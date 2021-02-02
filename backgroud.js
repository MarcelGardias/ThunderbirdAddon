const socket = new WebSocket('ws://127.0.0.2:5679');

// Connection opened
socket.addEventListener('open', function (event) {
    console.log('Connected to the WS Server!')
});
// Connection closed
socket.addEventListener('close', function (event) {
    console.log('Disconnected from the WS Server!')
});
// Listen for messages
socket.addEventListener('message', function (event) {
    console.log('Message from server ', event.data);
    var replyMessage = event.data;
    handleServerMessage(event.data);
});


// Error handling
socket.onerror = function(event) {
    console.error("WebSocket error observed:", event);
};

// Decisionmaking for data recieved from python 
function handleServerMessage(data){
    if (data == "error"){
        var sending = browser.runtime.sendMessage({
            process : "Error",
            message : "An error has occured. Check if all scripts are running and the required packaged are installed!"
          });
    }else{
        var sending = browser.runtime.sendMessage({
            process : "generatedEmail",
            email: data
          });
    }
}

// Message handling for script communication  in the Add-On
function handleMessage(request) {
    console.log("Message from the content script: " +
      request.process);
      if (request.process == "sendContent" ){
        //console.log("1")
        var sendDict = [{"email": request.emailText,
                        "type": request.type}]
        //socket.send(sendDict)
        console.log(request.emailText)
        socket.send(request.emailText)
      }else if(request.process == "reply"){
        //console.log("2")
        reply(request.emailContent);
      }else if(request.process == "replyAll"){
        //console.log("3")
        replyAll(request.emailContent);
      }
  }
// Calling the Listener for Communication between scripts in the Thunderbird Add-On
browser.runtime.onMessage.addListener(handleMessage);


async function reply(generatedReply){
    const tabs = await browser.tabs.query({
        active: true,
        currentWindow: true,
    })
    const tabId = tabs[0].id;
    const message = await browser.messageDisplay.getDisplayedMessage(tabId);
    const composer = await browser.compose.beginReply(message.id, "replyToSender");
    const details = await browser.compose.getComposeDetails(composer.id);
    //console.log(details);

    let newDetails = {};
    if (details.isPlainText) {
        newDetails.plainTextBody = generatedReply;
    } else {
        var generatedReplyHtml = "<b>"+ generatedReply +"</b>";
        newDetails.body = generatedReplyHtml;
    }
    await browser.compose.setComposeDetails(composer.id, newDetails);
}
async function replyAll(generatedReply){
    const tabs = await browser.tabs.query({
        active: true,
        currentWindow: true,
    })
    const tabId = tabs[0].id;
    const message = await browser.messageDisplay.getDisplayedMessage(tabId);
    const composer = await browser.compose.beginReply(message.id, "replyToAll");
    const details = await browser.compose.getComposeDetails(composer.id);
    console.log(details);

    let newDetails = {};
    if (details.isPlainText) {
        newDetails.plainTextBody = generatedReply;
    } else {
        var generatedReplyHtml = "<b>"+ generatedReply +"</b>";
        newDetails.body = generatedReplyHtml;
    }
    await browser.compose.setComposeDetails(composer.id, newDetails);
}