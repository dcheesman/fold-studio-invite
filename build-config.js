// Build script for Vercel deployment
// This script injects environment variables into config.js

const fs = require('fs');

// Read the template config.js
let config = fs.readFileSync('config.js', 'utf8');

// Replace placeholders with environment variables
config = config.replace('YOUR_API_KEY_HERE', process.env.AIRTABLE_API_KEY || 'YOUR_API_KEY_HERE');
config = config.replace('YOUR_BASE_ID_HERE', process.env.AIRTABLE_BASE_ID || 'YOUR_BASE_ID_HERE');

// Write the updated config.js
fs.writeFileSync('config.js', config);

console.log('✅ Environment variables injected into config.js');
