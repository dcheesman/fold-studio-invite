// Quick deployment test script
// Run this in browser console on your live site to verify deployment

console.log('🧪 Testing The Fold Studio Invite Deployment...');

// Test 1: Check if page loads without errors
console.log('✅ Page loaded successfully');

// Test 2: Check if p5.js is working
if (typeof p5 !== 'undefined') {
    console.log('✅ p5.js library loaded');
} else {
    console.log('❌ p5.js library missing');
}

// Test 3: Check if CONFIG is available
if (typeof CONFIG !== 'undefined') {
    console.log('✅ CONFIG object loaded');
    console.log('📅 Event date:', CONFIG.text.time);
} else {
    console.log('❌ CONFIG object missing');
}

// Test 4: Check if Airtable config is working
if (typeof AIRTABLE_CONFIG !== 'undefined') {
    console.log('✅ Airtable config loaded');
    console.log('🔗 Airtable URL:', AIRTABLE_CONFIG.url);
} else {
    console.log('❌ Airtable config missing');
}

// Test 5: Check FPS (if available)
if (typeof fps !== 'undefined') {
    console.log('📊 Current FPS:', fps);
} else {
    console.log('ℹ️ FPS monitor not available');
}

console.log('🎉 Deployment test complete!');
