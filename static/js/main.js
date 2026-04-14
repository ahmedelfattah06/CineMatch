/* CineMatch — Main JavaScript */

// Utility: sync toggle icon with current theme
function syncThemeIcon() {
    var isDark = document.documentElement.classList.contains('dark');
    var sun  = document.getElementById('iconSun');
    var moon = document.getElementById('iconMoon');
    if (sun)  sun.style.display  = isDark ? 'none'  : 'block';
    if (moon) moon.style.display = isDark ? 'block' : 'none';
}

document.addEventListener('DOMContentLoaded', function () {

    // Sync icon with whatever theme was applied by the <head> anti-FOUC script
    syncThemeIcon();

    // Mobile nav toggle
    var toggle = document.getElementById('navToggle');
    var links  = document.getElementById('navLinks');
    var backdrop = document.getElementById('navBackdrop');
    if (toggle && links) {
        toggle.addEventListener('click', function () {
            links.classList.toggle('open');
            if (backdrop) backdrop.classList.toggle('open');
        });
        // Close drawer when clicking backdrop
        if (backdrop) {
            backdrop.addEventListener('click', function () {
                links.classList.remove('open');
                backdrop.classList.remove('open');
            });
        }
        // Close drawer when clicking outside
        document.addEventListener('click', function (e) {
            if (!toggle.contains(e.target) && !links.contains(e.target) && !(backdrop && backdrop.contains(e.target))) {
                links.classList.remove('open');
                if (backdrop) backdrop.classList.remove('open');
            }
        });
    }

    // Theme toggle
    var themeBtn = document.getElementById('themeToggle');
    if (themeBtn) {
        themeBtn.addEventListener('click', function () {
            var isDark = document.documentElement.classList.contains('dark');
            if (isDark) {
                document.documentElement.classList.remove('dark');
                localStorage.setItem('theme', 'light');
            } else {
                document.documentElement.classList.add('dark');
                localStorage.setItem('theme', 'dark');
            }
            syncThemeIcon();
        });
    }
});
