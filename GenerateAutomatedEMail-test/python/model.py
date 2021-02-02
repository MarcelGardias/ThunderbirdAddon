import tensorflow as tf
from sklearn.model_selection import train_test_split
import unicodedata
import re
import numpy as np
import pandas as pd
import os
import time

class model():
    def __init__(self):
        super(model, self).__init__()

    # ============================================================
    # Converts the unicode file to ascii
    # ============================================================
    def unicode_to_ascii(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


    # ============================================================
    # Preprocess a given sentence
    # ============================================================
    def preprocess_sentence(self, w):
        w = self.unicode_to_ascii(w.lower().strip())

        # creating a space between a word and the punctuation following it
        w = re.sub(r"([?.!,])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)

        # replacing everything with space except (a-z, A-Z, 0-9, ".", "?", "!", ",")
        # w = re.sub(r"[^a-zA-Z0-9?.!,]+", " ", w)
        w = w.strip()

        # adding a start and an end token to the sentence
        w = '<start> ' + w + ' <end>'
        return w


    # ============================================================
    # Create an array containing the mails and replies
    # ============================================================
    def create_dataset(self, emails):
        body = []
        reply = []
        for i in range(len(emails)):
            s1 = str(emails.iloc[i]["question"])
            s2 = str(emails.iloc[i]["answer"])
            body.append(self.preprocess_sentence(s1))
            reply.append(self.preprocess_sentence(s2))
        return body, reply


    # ============================================================
    # Convert the text to a tensor
    # ============================================================
    def tokenize(self, lang):
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        lang_tokenizer.fit_on_texts(lang)
        tensor = lang_tokenizer.texts_to_sequences(lang)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
        max_length = tensor.shape[1]
        return tensor, lang_tokenizer, max_length


    # ============================================================
    # Split data in training and testing
    # ============================================================
    def split_data(self, inp_tensor, tar_tensor, t_size):
        input_train, input_val, target_train, target_val = train_test_split(inp_tensor, tar_tensor, test_size=t_size)
        return input_train, input_val, target_train, target_val


    # ============================================================
    # Encoder class
    # ============================================================
    class Encoder(tf.keras.Model):
        def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
            super(model.Encoder, self).__init__()
            self.batch_sz = batch_sz
            self.enc_units = enc_units
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
            self.gru = tf.keras.layers.GRU(self.enc_units,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_initializer='glorot_uniform')


        def call(self, x, hidden):
            x = self.embedding(x)
            output, state = self.gru(x, initial_state = hidden)
            return output, state

        def initialize_hidden_state(self):
            return tf.zeros((self.batch_sz, self.enc_units))


    # ============================================================
    # Attention class
    # ============================================================
    class BahdanauAttention(tf.keras.layers.Layer):
        def __init__(self, units):
            super(model.BahdanauAttention, self).__init__()
            self.W1 = tf.keras.layers.Dense(units)
            self.W2 = tf.keras.layers.Dense(units)
            self.V = tf.keras.layers.Dense(1)

        def call(self, query, values):
            # query hidden state shape == (batch_size, hidden size)
            # query_with_time_axis shape == (batch_size, 1, hidden size)
            # values shape == (batch_size, max_len, hidden size)
            # we are doing this to broadcast addition along the time axis to calculate the score
            query_with_time_axis = tf.expand_dims(query, 1)

            # score shape == (batch_size, max_length, 1)
            # we get 1 at the last axis because we are applying score to self.V
            # the shape of the tensor before applying self.V is (batch_size, max_length, units)
            score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))

            # attention_weights shape == (batch_size, max_length, 1)
            attention_weights = tf.nn.softmax(score, axis=1)

            # context_vector shape after sum == (batch_size, hidden_size)
            context_vector = attention_weights * values
            context_vector = tf.reduce_sum(context_vector, axis=1)
            return context_vector, attention_weights


    # ============================================================
    # Decoder class
    # ============================================================
    class Decoder(tf.keras.Model):
        def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
            super(model.Decoder, self).__init__()
            self.batch_sz = batch_sz
            self.dec_units = dec_units
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
            self.gru = tf.keras.layers.GRU(self.dec_units,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_initializer='glorot_uniform')
            self.fc = tf.keras.layers.Dense(vocab_size)
            self.attention = model.BahdanauAttention(self.dec_units)

        def call(self, x, hidden, enc_output):
            # enc_output shape == (batch_size, max_length, hidden_size)
            context_vector, attention_weights = self.attention(hidden, enc_output)

            # x shape after passing through embedding == (batch_size, 1, embedding_dim)
            x = self.embedding(x)

            # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
            x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

            # passing the concatenated vector to the GRU
            output, state = self.gru(x)

            # output shape == (batch_size * 1, hidden_size)
            output = tf.reshape(output, (-1, output.shape[2]))

            # output shape == (batch_size, vocab)
            x = self.fc(output)
            return x, state, attention_weights


    # ============================================================
    # Loss function
    # ============================================================
    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)


    # ============================================================
    # Training step
    # ============================================================
    @tf.function
    def train_step(self, inp, targ, enc_hidden):
        loss = 0
        print("initialize training.")

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder(inp, enc_hidden)
            print("enc_output and enc_hidden done.")
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([self.targ_lang_tok.word_index['<start>']] * self.BATCH_SIZE, 1)
            print("dec_input done.")

            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                print("target shape", t, "out of", targ.shape[1] - 1)
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
                loss += self.loss_function(targ[:, t], predictions)
                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss


    # ============================================================
    # Evaluate
    # ============================================================
    def evaluate(self, sentence):
        attention_plot = np.zeros((self.max_length_targ, self.max_length_inp))
        sentence = self.preprocess_sentence(sentence)
        inputs = [self.inp_lang_tok.word_index[i] for i in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=self.max_length_inp, padding='post')
        inputs = tf.convert_to_tensor(inputs)
        result = ''
        hidden = [tf.zeros((1, self.units))]
        enc_out, enc_hidden = self.encoder(inputs, hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.targ_lang_tok.word_index['<start>']], 0)

        for t in range(self.max_length_targ):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input, dec_hidden, enc_out)

            # storing the attention weights to plot later on
            attention_weights = tf.reshape(attention_weights, (-1,))
            attention_plot[t] = attention_weights.numpy()
            predicted_id = tf.argmax(predictions[0]).numpy()
            result += self.targ_lang_tok.index_word[predicted_id] + ' '

            if self.targ_lang_tok.index_word[predicted_id] == '<end>':
                return result, sentence, attention_plot

            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)
        return result, sentence, attention_plot


    # ============================================================
    # Generate answer
    # ============================================================
    def translate(self, sentence):
        result, sentence, attention_plot = self.evaluate(sentence)
        result = result[:32]
        # print('Input: %s' % (sentence))

        return result



    # ============================================================
    # run function to train the model
    # ============================================================
    def trainModel(self, settings):

        # # TF GPU FIX
        # config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
        # config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        # session = tf.compat.v1.Session(config=config)
        # tf.compat.v1.keras.backend.set_session(session)
        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TF to use only the CPU


        filepath = settings["filepath"]
        emails = pd.read_csv(filepath)
        # REDUCE DATA
        print("len data: {}".format(len(emails)))
        emails = emails[:settings["data_size"]]
        print("reduced data length: {}". format(len(emails)))
        print("start preprocessing. NOTE: This will take a while.")
        body, reply = self.create_dataset(emails)
        input_tensor, inp_lang_tok, max_length_inp = self.tokenize(body)
        target_tensor, self.targ_lang_tok, max_length_targ = self.tokenize(reply)
        input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = self.split_data(input_tensor,
                                                                                                  target_tensor, 0.2)

        BUFFER_SIZE = len(input_tensor_train)
        self.BATCH_SIZE = settings["BATCH_SIZE"]
        steps_per_epoch = len(input_tensor_train) // self.BATCH_SIZE
        self.embedding_dim = settings["embedding_dim"]
        self.units = settings["units"]
        self.vocab_inp_size = len(inp_lang_tok.word_index) + 1
        self.vocab_tar_size = len(self.targ_lang_tok.word_index) + 1

        dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
        dataset = dataset.batch(self.BATCH_SIZE, drop_remainder=True)

        example_input_batch, example_target_batch = next(iter(dataset))

        self.encoder = model.Encoder(self.vocab_inp_size, self.embedding_dim, self.units, self.BATCH_SIZE)

        sample_hidden = self.encoder.initialize_hidden_state()
        sample_output, sample_hidden = self.encoder(example_input_batch, sample_hidden)
        print("This is the last working row.")
        attention_layer = model.BahdanauAttention(10)
        attention_result, attention_weights = attention_layer(sample_hidden, sample_output)
        print("Program already crashed at this point.")

        self.decoder = model.Decoder(self.vocab_tar_size, self.embedding_dim, self.units, self.BATCH_SIZE)
        sample_decoder_output, _, _ = self.decoder(tf.random.uniform((self.BATCH_SIZE, 1)), sample_hidden, sample_output)

        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        checkpoint_dir = settings["checkpoint_dir"]
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, encoder=self.encoder, decoder=self.decoder)

        EPOCHS = settings["EPOCHS"]
        for epoch in range(EPOCHS):
            print(epoch, "out of", EPOCHS)
            start = time.time()

            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0

            for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
                # print(batch)
                batch_loss = self.train_step(inp, targ, enc_hidden)
                total_loss += batch_loss

                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
            # saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 2 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)
            print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))



    # ============================================================
    # run function to predict emails
    # ============================================================
    def predict(self, settings):
        # # TF GPU fix
        # config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
        # config.gpu_options.allow_growth = True
        # session = tf.compat.v1.Session(config=config)
        # tf.compat.v1.keras.backend.set_session(session)

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TF to use only the CPU

        checkpoint_dir = settings["checkpoint_dir"]
        email = settings["email"]
        self.BATCH_SIZE = settings["BATCH_SIZE"]
        self.embedding_dim = settings["embedding_dim"]
        self.units = settings["units"]
        filepath = settings["filepath"]
        emails = pd.read_csv(filepath)
        # LIMIT DATASIZE TO X
        emails = emails[:settings["data_size"]]
        body, reply = self.create_dataset(emails)
        # self.body = body

        input_tensor, inp_lang_tok, max_length_inp = self.tokenize(body)
        target_tensor, targ_lang_tok, max_length_targ = self.tokenize(reply)
        self.vocab_inp_size = len(inp_lang_tok.word_index) + 1
        self.vocab_tar_size = len(targ_lang_tok.word_index) + 1

        self.max_length_inp = max_length_inp
        self.max_length_targ = max_length_targ
        self.inp_lang_tok = inp_lang_tok
        self.targ_lang_tok = targ_lang_tok

        self.encoder = model.Encoder(self.vocab_inp_size, self.embedding_dim, self.units, self.BATCH_SIZE)
        self.decoder = model.Decoder(self.vocab_tar_size, self.embedding_dim, self.units, self.BATCH_SIZE)

        # ICH HOFF MAL DASS DAS SO FUNKTIONIERT DEN CHECKPOINT ZU LADEN UND DAS ZU EVALUIEREN
        checkpoint = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(), encoder=self.encoder, decoder=self.decoder)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

        string = settings["input"]
        return self.translate(string)




