// Road Monitor - API Helper
// Handles authentication and API calls

const API_BASE = window.ROADGUARD_API_BASE || (
    window.location.protocol.startsWith('http')
        ? `${window.location.origin}/api`
        : 'http://localhost:5000/api'
);

const API = {
    // Get stored token
    getToken() {
        return localStorage.getItem('token');
    },

    // Get auth headers
    getHeaders() {
        const headers = { 'Content-Type': 'application/json' };
        const token = this.getToken();
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }
        return headers;
    },

    getStreamUrl() {
        const token = this.getToken();
        const tokenParam = token ? `&token=${encodeURIComponent(token)}` : '';
        return `${API_BASE}/stream?t=${Date.now()}${tokenParam}`;
    },

    getEvidenceUrl(path) {
        if (!path) return '';
        const base = API_BASE.replace(/\/api$/, '');
        return `${base}/evidence/${encodeURIComponent(path)}`;
    },

    hasRequiredRole(requiredRole) {
        const token = this.getToken();
        const role = localStorage.getItem('role');
        if (!token || !role) return false;
        if (!requiredRole) return true;
        return role === requiredRole;
    },

    redirectForRole() {
        const role = localStorage.getItem('role');
        if (role === 'admin') {
            window.location.href = '../admin/dashboard.html';
        } else {
            window.location.href = '../index.html';
        }
    },

    // Login
    async login(emailOrUsername, password) {
        const res = await fetch(`${API_BASE}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                email: emailOrUsername,
                username: emailOrUsername,
                password
            })
        });
        const data = await res.json();
        if (data.ok && data.token) {
            localStorage.setItem('token', data.token);
            localStorage.setItem('role', data.role);
            localStorage.setItem('userId', data.user_id);
            if (data.name) localStorage.setItem('userName', data.name);
        }
        return data;
    },

    // Register
    async register(email, password, name) {
        const res = await fetch(`${API_BASE}/auth/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password, name })
        });
        const data = await res.json();
        if (data.ok && data.token) {
            localStorage.setItem('token', data.token);
            localStorage.setItem('role', data.role);
            localStorage.setItem('userId', data.user_id);
        }
        return data;
    },

    // Logout
    logout() {
        localStorage.removeItem('token');
        localStorage.removeItem('role');
        localStorage.removeItem('userId');
        localStorage.removeItem('userName');
        window.location.href = '../index.html';
    },

    // Get current user
    async getMe() {
        const res = await fetch(`${API_BASE}/auth/me`, {
            headers: this.getHeaders()
        });
        return res.json();
    },

    // ============ USER ENDPOINTS ============

    // Get user's vehicles
    async getUserVehicles() {
        const res = await fetch(`${API_BASE}/user/vehicles`, {
            headers: this.getHeaders()
        });
        return res.json();
    },

    // Add vehicle
    async addVehicle(vehicleData) {
        const res = await fetch(`${API_BASE}/user/vehicles`, {
            method: 'POST',
            headers: this.getHeaders(),
            body: JSON.stringify(vehicleData)
        });
        return res.json();
    },

    // Update vehicle
    async updateVehicle(id, vehicleData) {
        const res = await fetch(`${API_BASE}/user/vehicles/${id}`, {
            method: 'PUT',
            headers: this.getHeaders(),
            body: JSON.stringify(vehicleData)
        });
        return res.json();
    },

    // Get User Events (Safety Analysis)
    async getUserEvents(limit = null, type = null, evidenceOnly = false) {
        const params = new URLSearchParams();
        if (limit !== null && limit !== undefined) params.set('limit', String(limit));
        if (type) params.set('type', String(type));
        if (evidenceOnly) params.set('evidence_only', '1');
        const query = params.toString() ? `?${params.toString()}` : '';

        const res = await fetch(`${API_BASE}/user/events${query}`, {
            headers: this.getHeaders()
        });
        return res.json();
    },

    // Get Potholes Map Data
    async getPotholes() {
        const res = await fetch(`${API_BASE}/potholes`, {
            headers: this.getHeaders()
        });
        return res.json();
    },

    // Get Pothole Events (up to 200, for analytics page)
    async getPotholeEvents() {
        const res = await fetch(`${API_BASE}/user/events?type=pothole&limit=200`, {
            headers: this.getHeaders()
        });
        return res.json();
    },

    // Get Hazard Records (dedicated hazard DB)
    async getHazards(limit = 200) {
        const res = await fetch(`${API_BASE}/hazards?limit=${encodeURIComponent(limit)}`, {
            headers: this.getHeaders()
        });
        return res.json();
    },

    // Get Behavior History
    async getBehaviorHistory() {
        const res = await fetch(`${API_BASE}/behavior/history`, {
            headers: this.getHeaders()
        });
        return res.json();
    },

    // Upload Video
    async uploadVideo(file) {
        const formData = new FormData();
        formData.append('file', file);

        const res = await fetch(`${API_BASE}/upload/video`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${this.getToken()}`
            },
            body: formData
        });
        return res.json();
    },

    // Start System (Live or Video)
    async startSystem(mode, source = 0) {
        const res = await fetch(`${API_BASE}/system/start`, {
            method: 'POST',
            headers: this.getHeaders(),
            body: JSON.stringify({ mode, source })
        });
        return res.json();
    },

    // Stop System
    async stopSystem() {
        const res = await fetch(`${API_BASE}/system/stop`, {
            method: 'POST',
            headers: this.getHeaders()
        });
        return res.json();
    },

    // Upload Video with Vehicle ID
    async uploadVideoWithVehicle(file, vehicleId) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('vehicle_id', vehicleId || 'DEMO-CAR-01');

        const res = await fetch(`${API_BASE}/upload/video`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${this.getToken()}`
            },
            body: formData
        });
        return res.json();
    },

    // Delete vehicle
    async deleteVehicle(id) {
        const res = await fetch(`${API_BASE}/user/vehicles/${id}`, {
            method: 'DELETE',
            headers: this.getHeaders()
        });
        return res.json();
    },

    // Get violation events for a specific vehicle
    async getVehicleEvents(vehicleId) {
        const res = await fetch(`${API_BASE}/user/vehicles/${vehicleId}/events`, {
            headers: this.getHeaders()
        });
        return res.json();
    },

    // ============ ADMIN ENDPOINTS ============

    // Get system summary (admin dashboard)
    async getSummary() {
        const res = await fetch(`${API_BASE}/summary`, {
            headers: this.getHeaders()
        });
        return res.json();
    },

    // Get all users (admin)
    async getAdminUsers() {
        const res = await fetch(`${API_BASE}/admin/users`, {
            headers: this.getHeaders()
        });
        return res.json();
    },

    // Get all vehicles (admin)
    async getAdminVehicles() {
        const res = await fetch(`${API_BASE}/admin/vehicles`, {
            headers: this.getHeaders()
        });
        return res.json();
    },

    // Delete user (admin)
    async deleteUser(id) {
        const res = await fetch(`${API_BASE}/admin/users/${id}`, {
            method: 'DELETE',
            headers: this.getHeaders()
        });
        return res.json();
    },

    // Save calibration result to a vehicle (admin)
    async saveCalibration(vehicleId, kmhPerPixel) {
        const res = await fetch(`${API_BASE}/admin/vehicles/${vehicleId}/calibrate`, {
            method: 'POST',
            headers: this.getHeaders(),
            body: JSON.stringify({ kmh_per_pixel: kmhPerPixel })
        });
        return res.json();
    },

    // Add vehicle (admin)
    async addAdminVehicle(data) {
        const res = await fetch(`${API_BASE}/admin/vehicles`, {
            method: 'POST',
            headers: this.getHeaders(),
            body: JSON.stringify(data)
        });
        return res.json();
    },

    // Get single vehicle (admin)
    async getAdminVehicle(id) {
        const res = await fetch(`${API_BASE}/admin/vehicles/${id}`, {
            headers: this.getHeaders()
        });
        return res.json();
    },

    // Update vehicle (admin)
    async updateAdminVehicle(id, data) {
        const res = await fetch(`${API_BASE}/admin/vehicles/${id}`, {
            method: 'PUT',
            headers: this.getHeaders(),
            body: JSON.stringify(data)
        });
        return res.json();
    },

    // Delete vehicle (admin)
    async deleteAdminVehicle(id) {
        const res = await fetch(`${API_BASE}/admin/vehicles/${id}`, {
            method: 'DELETE',
            headers: this.getHeaders()
        });
        return res.json();
    },

    // Get violation events for a vehicle (admin — any vehicle)
    async getAdminVehicleEvents(id) {
        const res = await fetch(`${API_BASE}/admin/vehicles/${id}/events`, {
            headers: this.getHeaders()
        });
        return res.json();
    }
};

function requireAuth(requiredRole) {
    const path = window.location.pathname.replace(/\\/g, '/');
    const fallback = path.includes('/admin/') ? '../index.html' : '../index.html';

    if (!API.hasRequiredRole(requiredRole)) {
        if (requiredRole === 'user' && localStorage.getItem('role') === 'admin') {
            window.location.href = '../admin/dashboard.html';
        } else {
            window.location.href = fallback;
        }
        return false;
    }
    return true;
}

(function guardProtectedPages() {
    const path = window.location.pathname.replace(/\\/g, '/');
    if (path.includes('/admin/')) {
        requireAuth('admin');
    } else if (path.includes('/user/')) {
        requireAuth('user');
    }
})();

function normalizeUserSidebar() {
    const path = window.location.pathname.replace(/\\/g, '/');
    if (!path.includes('/user/')) return;

    const current = path.split('/').pop() || 'dashboard.html';
    const active = 'flex items-center gap-3 px-3 py-2 rounded-lg bg-indigo-600/10 text-indigo-400';
    const inactive = 'flex items-center gap-3 px-3 py-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition';
    const icon = {
        dashboard: '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>',
        realtime: '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.723v6.554a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path>',
        analysis: '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z"></path>',
        sensors: '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>',
        vehicles: '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 17a2 2 0 11-4 0 2 2 0 014 0zM19 17a2 2 0 11-4 0 2 2 0 014 0z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16V6a1 1 0 00-1-1H4a1 1 0 00-1 1v10l2 2h6l2-2zm0 0h4a1 1 0 001-1v-4l-3-5h-2"></path>',
        map: '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7"></path>'
    };
    const items = [
        ['dashboard.html', 'Risk Dashboard', icon.dashboard],
        ['realtime.html', 'Live Detection', icon.realtime],
        ['analysis.html', 'Analysis & Upload', icon.analysis],
        ['sensors.html', 'Driver Behavior', icon.sensors],
        ['vehicles.html', 'My Vehicles', icon.vehicles],
        ['pothole.html', 'Pothole Analytics', icon.map],
        ['hazard.html', 'Hazard Analytics', icon.map]
    ];

    const nav = document.querySelector('aside nav');
    if (!nav) return;
    nav.innerHTML = items.map(([href, label, svg]) => `
        <a href="${href}" class="${current === href ? active : inactive}">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">${svg}</svg>
            <span${current === href ? ' class="font-medium"' : ''}>${label}</span>
        </a>
    `).join('');
}

function setupMobileSidebar() {
    const path = window.location.pathname.replace(/\\/g, '/');
    if (!path.includes('/user/') && !path.includes('/admin/')) return;
    const aside = document.querySelector('aside');
    if (!aside || document.getElementById('mobileNavToggle')) return;

    const btn = document.createElement('button');
    btn.id = 'mobileNavToggle';
    btn.type = 'button';
    btn.className = 'mobile-nav-toggle lg:hidden';
    btn.setAttribute('aria-label', 'Open navigation');
    btn.innerHTML = `
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 7h16M4 12h16M4 17h16"></path>
        </svg>
    `;
    btn.addEventListener('click', () => {
        const open = document.body.classList.toggle('mobile-nav-open');
        btn.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
    });
    document.body.appendChild(btn);

    aside.addEventListener('click', (event) => {
        const link = event.target.closest('a');
        if (link) document.body.classList.remove('mobile-nav-open');
    });

    document.addEventListener('click', (event) => {
        if (!document.body.classList.contains('mobile-nav-open')) return;
        if (aside.contains(event.target) || btn.contains(event.target)) return;
        document.body.classList.remove('mobile-nav-open');
    });
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        normalizeUserSidebar();
        setupMobileSidebar();
    });
} else {
    normalizeUserSidebar();
    setupMobileSidebar();
}
