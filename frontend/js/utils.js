// Road Monitor - Utility Functions

// Auth guard — redirects to login if not authenticated
function requireAuth(requiredRole) {
    const token = localStorage.getItem('token');
    const role = localStorage.getItem('role');
    if (!token) {
        window.location.href = '../index.html';
        return false;
    }
    if (requiredRole && role !== requiredRole) {
        window.location.href = '../index.html';
        return false;
    }
    return true;
}

// Toast Notifications
const Toast = {
    container: null,

    init() {
        if (!this.container) {
            this.container = document.createElement('div');
            this.container.className = 'toast-container';
            document.body.appendChild(this.container);
        }
    },

    show(message, type = 'success', duration = 3000) {
        this.init();

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;

        const icon = type === 'success' ? '✓' : type === 'error' ? '✕' : '!';
        toast.innerHTML = `
      <span class="text-lg">${icon}</span>
      <span class="text-sm text-slate-200">${message}</span>
    `;

        this.container.appendChild(toast);

        setTimeout(() => {
            toast.style.animation = 'slideIn 0.3s ease reverse';
            setTimeout(() => toast.remove(), 300);
        }, duration);
    },

    success(message) { this.show(message, 'success'); },
    error(message) { this.show(message, 'error'); },
    warning(message) { this.show(message, 'warning'); }
};

// Modal Management
const Modal = {
    current: null,

    open(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.add('active');
            this.current = modal;
        }
    },

    close(modalId) {
        const modal = modalId ? document.getElementById(modalId) : this.current;
        if (modal) {
            modal.classList.remove('active');
            this.current = null;
        }
    },

    // Confirm dialog
    confirm(message, title = 'Confirm') {
        return new Promise((resolve) => {
            const overlay = document.createElement('div');
            overlay.className = 'modal-overlay active';
            overlay.innerHTML = `
        <div class="modal-content">
          <h3 class="text-lg font-semibold text-white mb-2">${title}</h3>
          <p class="text-slate-400 mb-6">${message}</p>
          <div class="flex gap-3 justify-end">
            <button class="btn btn-secondary" id="confirmCancel">Cancel</button>
            <button class="btn btn-danger" id="confirmOk">Delete</button>
          </div>
        </div>
      `;

            document.body.appendChild(overlay);

            overlay.querySelector('#confirmCancel').onclick = () => {
                overlay.remove();
                resolve(false);
            };

            overlay.querySelector('#confirmOk').onclick = () => {
                overlay.remove();
                resolve(true);
            };

            overlay.onclick = (e) => {
                if (e.target === overlay) {
                    overlay.remove();
                    resolve(false);
                }
            };
        });
    }
};

// Format date
function formatDate(dateStr) {
    if (!dateStr) return '-';
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// Format date short
function formatDateShort(dateStr) {
    if (!dateStr) return '-';
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// Violation type badge
function getViolationBadge(type) {
    if (type === 'lane_deviation') {
        return '<span class="badge badge-danger">Lane Violation</span>';
    } else if (type === 'overspeed') {
        return '<span class="badge badge-warning">Overspeed</span>';
    }
    return `<span class="badge badge-info">${type}</span>`;
}

// Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Loading state
function setLoading(element, loading) {
    if (loading) {
        element.disabled = true;
        element.dataset.originalText = element.innerHTML;
        element.innerHTML = '<span class="spinner"></span>';
    } else {
        element.disabled = false;
        element.innerHTML = element.dataset.originalText || element.innerHTML;
    }
}

// Debounce function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}
