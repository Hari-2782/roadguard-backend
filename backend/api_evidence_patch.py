from flask import send_from_directory, jsonify # Ensure jsonify is imported if not already

# ... existing code ...

# Add this route to api.py to serve evidence
@app.route('/evidence/<path:filename>')
def serve_evidence(filename):
    return send_from_directory(EVIDENCE_DIR, filename)
