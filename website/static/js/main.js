/**
 * Main JavaScript — Theme toggle, mobile nav, smooth scroll, and nav behavior.
 */
(function () {
    'use strict';

    // ════════════════════════════════════════════
    // THEME TOGGLE
    // ════════════════════════════════════════════
    const themeToggle = document.getElementById('theme-toggle');
    const html = document.documentElement;

    function setTheme(theme) {
        html.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);

        // Toggle sun/moon icons
        const sun = themeToggle.querySelector('.theme-toggle__sun');
        const moon = themeToggle.querySelector('.theme-toggle__moon');
        if (sun && moon) {
            sun.style.display = theme === 'dark' ? 'block' : 'none';
            moon.style.display = theme === 'light' ? 'block' : 'none';
        }
    }

    // Initialize theme from localStorage or default to dark
    const savedTheme = localStorage.getItem('theme') || 'dark';
    setTheme(savedTheme);

    if (themeToggle) {
        themeToggle.addEventListener('click', function () {
            const current = html.getAttribute('data-theme');
            setTheme(current === 'dark' ? 'light' : 'dark');
        });
    }

    // ════════════════════════════════════════════
    // MOBILE NAVIGATION
    // ════════════════════════════════════════════
    const navToggle = document.getElementById('nav-toggle');
    const navLinks = document.getElementById('nav-links');

    if (navToggle && navLinks) {
        navToggle.addEventListener('click', function () {
            navLinks.classList.toggle('is-open');
        });

        // Close mobile nav when a link is clicked
        navLinks.querySelectorAll('.nav__link').forEach(function (link) {
            link.addEventListener('click', function () {
                navLinks.classList.remove('is-open');
            });
        });
    }

    // ════════════════════════════════════════════
    // ACCORDION
    // ════════════════════════════════════════════
    document.querySelectorAll('.accordion__trigger').forEach(function (trigger) {
        trigger.addEventListener('click', function () {
            const content = this.nextElementSibling;
            const isOpen = content.classList.contains('is-open');

            // Close all accordions in the same parent
            const parent = this.closest('.accordion').parentElement;
            parent.querySelectorAll('.accordion__content').forEach(function (c) {
                c.classList.remove('is-open');
            });
            parent.querySelectorAll('.accordion__trigger').forEach(function (t) {
                t.setAttribute('aria-expanded', 'false');
            });

            // Toggle this one
            if (!isOpen) {
                content.classList.add('is-open');
                this.setAttribute('aria-expanded', 'true');
            }
        });
    });

    // ════════════════════════════════════════════
    // NAV SCROLL BEHAVIOR
    // ════════════════════════════════════════════
    const nav = document.getElementById('main-nav');
    if (nav) {
        let lastScroll = 0;
        window.addEventListener('scroll', function () {
            const currentScroll = window.pageYOffset;

            if (currentScroll > 100) {
                nav.style.boxShadow = 'var(--shadow-md)';
            } else {
                nav.style.boxShadow = 'none';
            }

            lastScroll = currentScroll;
        }, { passive: true });
    }

    // ════════════════════════════════════════════
    // DOCS SIDEBAR ACTIVE STATE
    // ════════════════════════════════════════════
    const docsLinks = document.querySelectorAll('.docs-nav__link');
    if (docsLinks.length > 0) {
        const observer = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                if (entry.isIntersecting) {
                    const id = entry.target.getAttribute('id');
                    docsLinks.forEach(function (link) {
                        link.classList.remove('docs-nav__link--active');
                        if (link.getAttribute('href') === '#' + id) {
                            link.classList.add('docs-nav__link--active');
                        }
                    });
                }
            });
        }, { rootMargin: '-20% 0px -70% 0px' });

        document.querySelectorAll('.docs-content h2[id]').forEach(function (heading) {
            observer.observe(heading);
        });
    }

})();
