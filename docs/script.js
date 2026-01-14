// Smooth scroll for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        const href = this.getAttribute('href');
        if (href === '#') return;
        
        e.preventDefault();
        const target = document.querySelector(href);
        if (target) {
            const navHeight = document.querySelector('.navbar').offsetHeight;
            const targetTop = target.offsetTop - navHeight;
            window.scrollTo({
                top: targetTop,
                behavior: 'smooth'
            });
        }
    });
});

// Add scroll effect to navbar
window.addEventListener('scroll', function() {
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 10) {
        navbar.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.15)';
    } else {
        navbar.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.1)';
    }
});

// Animate numbers in statistics section
function animateCounter(element, target, duration = 2000) {
    let current = 0;
    const increment = target / (duration / 16);
    
    const counter = setInterval(() => {
        current += increment;
        if (current >= target) {
            element.textContent = target;
            clearInterval(counter);
        } else {
            element.textContent = Math.floor(current);
        }
    }, 16);
}

// Trigger animations when stats section comes into view
const observerOptions = {
    threshold: 0.5,
    rootMargin: '0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const statNumbers = entry.target.querySelectorAll('.stat-number');
            statNumbers.forEach(num => {
                const value = parseInt(num.textContent);
                if (!isNaN(value)) {
                    animateCounter(num, value);
                }
            });
            observer.unobserve(entry.target);
        }
    });
}, observerOptions);

const statsSection = document.querySelector('.hero-stats');
if (statsSection) {
    observer.observe(statsSection);
}

// Add animation to cards on scroll
const cardObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry, index) => {
        if (entry.isIntersecting) {
            entry.target.style.animation = `fadeInUp 0.6s ease forwards ${index * 0.1}s`;
        }
    });
}, {
    threshold: 0.1
});

document.querySelectorAll('.feature-card, .doc-card, .stats-card, .algo-group').forEach(card => {
    cardObserver.observe(card);
});

// CSS animation for fade-in-up
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
`;
document.head.appendChild(style);

// Copy code functionality
function setupCodeCopy() {
    const codeBlocks = document.querySelectorAll('pre');
    codeBlocks.forEach(block => {
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.textContent = '📋 Copy';
        button.style.cssText = `
            position: absolute;
            top: 10px;
            right: 10px;
            padding: 5px 10px;
            background: #333;
            color: #00ff00;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            z-index: 10;
        `;
        
        button.addEventListener('click', () => {
            const code = block.querySelector('code').textContent;
            navigator.clipboard.writeText(code).then(() => {
                button.textContent = '✅ Copied!';
                setTimeout(() => {
                    button.textContent = '📋 Copy';
                }, 2000);
            });
        });
        
        block.style.position = 'relative';
        block.appendChild(button);
    });
}

// Initialize code copy
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupCodeCopy);
} else {
    setupCodeCopy();
}

// Search functionality (if search feature is added)
function initSearch() {
    const searchInput = document.querySelector('.search-input');
    if (!searchInput) return;
    
    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase();
        const cards = document.querySelectorAll('[data-searchable]');
        
        cards.forEach(card => {
            const text = card.textContent.toLowerCase();
            if (text.includes(query) || query === '') {
                card.style.display = 'block';
            } else {
                card.style.display = 'none';
            }
        });
    });
}

initSearch();

// Mobile menu toggle
function setupMobileMenu() {
    const menuToggle = document.querySelector('.menu-toggle');
    const navMenu = document.querySelector('.nav-menu');
    
    if (menuToggle && navMenu) {
        menuToggle.addEventListener('click', () => {
            navMenu.classList.toggle('active');
        });
        
        // Close menu when a link is clicked
        navMenu.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                navMenu.classList.remove('active');
            });
        });
    }
}

setupMobileMenu();

// Keyboard navigation
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        const navMenu = document.querySelector('.nav-menu');
        if (navMenu) {
            navMenu.classList.remove('active');
        }
    }
});

// Print page statistics
console.log('%c🚀 Welcome to DEAP v4.0 - QD-NAS', 'font-size: 20px; color: #1f77d4; font-weight: bold;');
console.log('%cDistributed Evolutionary Algorithms Platform', 'font-size: 14px; color: #666;');
console.log('%c📊 Project Statistics:', 'font-size: 12px; color: #1f77d4; font-weight: bold;');
console.log('• 65+ Python files');
console.log('• 9000+ lines of core code');
console.log('• 25+ test functions');
console.log('• 12+ core algorithms');
console.log('• 15000+ lines of documentation');
console.log('• 50+ code examples');
console.log('%c🔗 Quick Links:', 'font-size: 12px; color: #1f77d4; font-weight: bold;');
console.log('GitHub: https://github.com/johboby/deap--qd--nas');
console.log('Documentation: See DOCS_INDEX.md');
console.log('Contributing: See CONTRIBUTING.md');
