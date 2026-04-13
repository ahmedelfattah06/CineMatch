/* CineMatch — Main JavaScript */

// Mobile nav toggle
document.addEventListener('DOMContentLoaded', function () {
    const toggle = document.getElementById('navToggle');
    const links = document.getElementById('navLinks');
    if (toggle && links) {
        toggle.addEventListener('click', function () {
            links.classList.toggle('open');
        });
        // Close on outside click
        document.addEventListener('click', function (e) {
            if (!toggle.contains(e.target) && !links.contains(e.target)) {
                links.classList.remove('open');
            }
        });
    }
});
