async function sendMessage(){

let input=document.getElementById("user-input");
let chatBox=document.getElementById("chat-box");

let message=input.value.trim();

if(message==="") return;

chatBox.innerHTML+=`
<div class="message user">${message}</div>
`;

input.value="";

let response=await fetch("/chat",{
method:"POST",
headers:{
"Content-Type":"application/json"
},
body:JSON.stringify({message:message})
});

let data=await response.json();

chatBox.innerHTML+=`
<div class="message bot">${data.response}</div>
`;

chatBox.scrollTop=chatBox.scrollHeight;

}

document.getElementById("user-input")
.addEventListener("keypress",function(e){

if(e.key==="Enter"){
sendMessage();
}

});