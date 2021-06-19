
  // Your web app's Firebase configuration
  var firebaseConfig = {
    apiKey: "AIzaSyDw94BPXJU1sXGJo0CuArudotyn03YeRRk",
    authDomain: "audiorecognition-3590f.firebaseapp.com",
    projectId: "audiorecognition-3590f",
    storageBucket: "audiorecognition-3590f.appspot.com",
    messagingSenderId: "871173130152",
    appId: "1:871173130152:web:cff0bd11325bc6398fbf9d"
  };
  // Initialize Firebase
  firebase.initializeApp(firebaseConfig);

document.getElementById('signIn').addEventListener('submit', function(e){
  e.preventDefault();
  //getting user info
  var email = document.getElementById('adminID');
  var password = document.getElementById('password'); 
  var email = document.getElementById('adminID');
  if(email==="" || password===""){
    alert("Enter all field!");
  }
  else{
    firebase.auth().signInWithEmailAndPassword(email.value, password.value)
    .then(function(response){
      firebase.auth().onAuthStateChanged(user => {
        if(user){
          window.location = "profile.html";
        }
        else{
          alert("Wrong Id or password!");
        }
      });
    }).catch(function(error){
      alert(" Incorrect ID or password!");
      var errorCode = error.code;
      var errorMessage = error.message;
      console.log(errorCode, errorMessage);
    })
    
  }
});