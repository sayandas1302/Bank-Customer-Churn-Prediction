function updateSliderValue(sliderId, valueId) {
    var slider = document.getElementById(sliderId);
    var value = document.getElementById(valueId);
    value.textContent = slider.value;
}