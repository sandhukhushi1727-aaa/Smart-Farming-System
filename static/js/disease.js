const leafImage = document.getElementById("leafImage");
const imagePreview = document.getElementById("imagePreview");
const previewContainer = document.getElementById("previewContainer");
const imageName = document.getElementById("imageName");
const uploadPrompt = document.getElementById("uploadPrompt");
const uploadBox = document.getElementById("uploadBox");
const removeImage = document.getElementById("removeImage");
const diseaseForm = document.getElementById("diseaseForm");

let previewURL = null;

// Display the selected image before submission.
leafImage.addEventListener("change", function () {

```
const file = this.files[0];

if (!file) {
    clearPreview();
    return;
}

const allowedTypes = [
    "image/jpeg",
    "image/png",
    "image/webp"
];

// Check the selected file type.
if (!allowedTypes.includes(file.type)) {
    alert("Please select a JPG, JPEG, PNG, or WEBP image.");
    clearPreview();
    return;
}

// Limit the file size to 5 MB.
const maximumSize = 5 * 1024 * 1024;

if (file.size > maximumSize) {
    alert("Please select an image smaller than 5 MB.");
    clearPreview();
    return;
}

// Remove the previous preview URL if one exists.
if (previewURL) {
    URL.revokeObjectURL(previewURL);
}

// Create a temporary URL for the selected image.
previewURL = URL.createObjectURL(file);

// Display the image immediately.
imagePreview.src = previewURL;
imageName.textContent = file.name;

previewContainer.hidden = false;
uploadPrompt.hidden = true;
uploadBox.classList.add("has-image");
```

});

// Clear the selected image and preview.
function clearPreview() {

```
if (previewURL) {
    URL.revokeObjectURL(previewURL);
    previewURL = null;
}

leafImage.value = "";
imagePreview.removeAttribute("src");
imageName.textContent = "";

previewContainer.hidden = true;
uploadPrompt.hidden = false;
uploadBox.classList.remove("has-image");
```

}

// Remove Image button.
removeImage.addEventListener("click", function () {
clearPreview();
});

// Basic browser-side check before form submission.
diseaseForm.addEventListener("submit", function (event) {

```
if (!leafImage.files.length) {
    event.preventDefault();
    alert("Please select a plant leaf image first.");
}
```

});
