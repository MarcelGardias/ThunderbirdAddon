window.addEventListener("load", onLoad);
async function onLoad() {
    window.document.getElementById("trainmodel").addEventListener("click",trainModel);
    window.document.getElementById("predict").addEventListener("click",sendContentMail);
    browser.runtime.onMessage.addListener(handleMessageBackground);
    }

async function sendContentMail(){
    document.getElementById("start").innerHTML = "";
    document.getElementById("processing").innerHTML = "Your email is being processed...";
    const tabs = await browser.tabs.query({
        active: true,
        currentWindow: true,
        })
    const tabId = tabs[0].id;
    const message = await browser.messageDisplay.getDisplayedMessage(tabId);
    const messageraw = await browser.messages.getFull(message.id);
    var email = iterateMail(messageraw);
    var emailjson = {
        email:email[0],
        type:email[1]
        };

    var sending = browser.runtime.sendMessage({
        process : "sendContent",
        emailText: emailjson.email,
        type: emailjson.type
        });
    };

async function reply(){
    var sending = browser.runtime.sendMessage({
        process : "reply",
        emailContent : document.getElementById("email-content").innerHTML
      });
    }
async function replyAll(){
    var sending = browser.runtime.sendMessage({
        process : "replyAll",
        emailContent : document.getElementById("email-content").innerHTML
        });
    }
async function trainModel(){
        messageList = await browser.messages.list(type = "inbox")
        print(messageList)
    var sending = browser.runtime.sendMessage({
        process : "trainmodel",
        emailContent : inbox
        });
    }

function handleMessageBackground(request) {
    console.log("Message from the content script: " +
      request.process);
      if (request.process == "generatedEmail"){
        document.getElementById("processing").innerHTML = "";  
        document.getElementById("email-content").innerHTML = request.email;
        document.getElementById("header-email").innerHTML = "Your response:";
        window.document.getElementById("reply").addEventListener("click",reply);
        window.document.getElementById("replyAll").addEventListener("click",replyAll);    
      }else if(request.process == "Error"){
        document.getElementById("email-content").innerHTML = request.message;
      }
      }



function iterateMail(message){
    var finalmessage = "";
    for (i = 0; i < message.parts.length; i++) {
        if(message.parts[i].contentType == "text/html"){
            finalmessage = message.parts[i];
            console.log("HTML");
        }
        else if( message.parts[i].contentType == "text/plain"){
            finalmessage = message.parts[i];
            break;
        }
        else{
            return iterateMail(message.parts[i]);
        }
    }
    return [finalmessage.body, finalmessage.contentType];
    }