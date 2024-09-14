<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>3D Text with Webcam Background</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 0;
      padding: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100vh;
      background-color: #000; /* Set a dark background for better webcam visibility */
      perspective: 1500px; /* Increase perspective to enhance 3D effect */
      overflow: hidden; /* Prevents scrollbars */
    }
    #videoBackground {
      position: absolute;
      top: 0;
      left: 0;
      width: 100vw; /* Full viewport width */
      height: 100vh; /* Full viewport height */
      object-fit: cover; /* Ensures the video covers the background without distortion */
      z-index: -1; /* Places the video behind other elements */
    }
    #textContainer {
      margin-top: 20px;
      font-size: 48px; /* Increased font size for larger text */
      color: red;
      font-weight: bold;
      text-shadow: 1px 1px 0 rgba(0,0,0,0.2), 2px 2px 0 rgba(0,0,0,0.2),
                   3px 3px 0 rgba(0,0,0,0.2), 4px 4px 0 rgba(0,0,0,0.2);
      transform: rotateX(20deg); /* Adjust the rotation for a 3D effect */
      z-index: 1; /* Ensures the text appears above the video */
    }
    label {
      font-size: 24px; /* Increased label font size */
      margin-right: 10px;
    }
    input {
      font-size: 24px; /* Increased input font size */
      padding: 10px;
      border: 2px solid #ccc;
      border-radius: 4px;
    }
  </style>
</head>
<body>
  <!-- Webcam Feed Background -->
  <video id="videoBackground" autoplay muted></video>

  <!-- Input Field for User to Enter Text -->
  <div style="display: flex; align-items: center;">
    <label for="userText">Enter Text:</label>
    <input
      type="text"
      id="userText"
      placeholder="Type something..."
      oninput="updateText()"
    />
  </div>

  <!-- Display Updated Text -->
  <div id="textContainer">Hello World!</div>

  <!-- JavaScript to Update Text and Access Webcam -->
  <script>
    // Function to update the text in the text container
    function updateText() {
      var inputText = document.getElementById("userText").value;
      var textContainer = document.getElementById("textContainer");
      textContainer.textContent = inputText;
    }

    // Function to start the webcam feed
    async function startWebcam() {
      try {
        // Access the webcam
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        // Set the video source to the webcam feed
        document.getElementById('videoBackground').srcObject = stream;
      } catch (err) {
        console.error('Error accessing webcam: ', err);
      }
    }

    // Start the webcam feed when the page loads
    startWebcam();
  </script>
</body>
</html>