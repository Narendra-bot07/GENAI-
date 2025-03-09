document.addEventListener('DOMContentLoaded', function () {
    const flightInfo = document.getElementById('flight-info');
    const services = document.getElementById('services');
    const baggageClaim = document.getElementById('baggage-claim');
    const airportMap = document.getElementById('airport-map');

    // Display detailed information when an item is clicked
    function showInfo(item) {
        alert(`You selected: ${item}`);
    }

    // Add event listeners to menu items
    flightInfo.addEventListener('click', function () {
        showInfo('Flight Information');
    });

    services.addEventListener('click', function () {
        showInfo('Services');
    });

    baggageClaim.addEventListener('click', function () {
        showInfo('Baggage Claim');
    });

    airportMap.addEventListener('click', function () {
        showInfo('Airport Map');
    });
});
