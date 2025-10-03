// Configuration file for The Fold Studio Grand Opening
// Copy this file to config.js and replace with your actual Airtable credentials

const AIRTABLE_CONFIG = {
    // Your Airtable Base ID (found in your base URL)
    baseId: 'YOUR_BASE_ID_HERE',
    
    // Your Airtable API Key (generate from https://airtable.com/create/tokens)
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
