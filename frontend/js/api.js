// Road Monitor - API Helper
// Handles authentication and API calls

const API_BASE = 'http://localhost:5000/api';

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
    async getUserEvents(limit = null, type = null) {
        const params = new URLSearchParams();
        if (limit !== null && limit !== undefined) params.set('limit', String(limit));
        if (type) params.set('type', String(type));
        const query = params.toString() ? `?${params.toString()}` : '';

        const res = await fetch(`${API_BASE}/user/events${query}`, {
            headers: this.getHeaders()
        });
        return res.json();
    },

    // Get Potholes Map Data
    async getPotholes() {
        // Public endpoint
        const res = await fetch(`${API_BASE}/potholes`);
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
