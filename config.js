// Configuration file for The Fold Studio Grand Opening
// Safe values for production deployment

const AIRTABLE_CONFIG = {
    // Your Airtable Base ID (found in your base URL)
    baseId: 'appCqw66jSnW2SnDC',
    
    // Your Airtable API Key (set by Vercel environment variables in production)
    apiKey: 'YOUR_API_KEY_HERE',
    
    // Your table name (should be "RSVPs")
    tableName: 'RSVPs',
    
    // Airtable API endpoint
    get url() {
        return `https://api.airtable.com/v0/${this.baseId}/${this.tableName}`;
    }
};

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AIRTABLE_CONFIG;
}