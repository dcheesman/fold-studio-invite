// Build script for Vercel deployment
// This script injects environment variables into config.js

const fs = require('fs');

console.log('🔧 Starting config.js build process...');
console.log('Environment variables available:');
console.log('AIRTABLE_BASE_ID:', process.env.AIRTABLE_BASE_ID ? 'SET' : 'NOT SET');
console.log('AIRTABLE_API_KEY:', process.env.AIRTABLE_API_KEY ? 'SET' : 'NOT SET');

// Read the template config.js
let config = fs.readFileSync('config.js', 'utf8');

// Replace placeholders with environment variables
const baseId = process.env.AIRTABLE_BASE_ID || 'YOUR_BASE_ID_HERE';
const apiKey = process.env.AIRTABLE_API_KEY || 'YOUR_API_KEY_HERE';

config = config.replace('YOUR_API_KEY_HERE', apiKey);
config = config.replace('YOUR_BASE_ID_HERE', baseId);

// Write the updated config.js
fs.writeFileSync('config.js', config);

console.log('✅ Environment variables injected into config.js');
console.log('Final baseId:', baseId);
console.log('Final apiKey:', apiKey.substring(0, 10) + '...');
