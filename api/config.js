// Vercel serverless function to serve config with environment variables
module.exports = (req, res) => {
  console.log('🔧 Config API called');
  console.log('Environment variables:');
  console.log('AIRTABLE_BASE_ID:', process.env.AIRTABLE_BASE_ID ? 'SET' : 'NOT SET');
  console.log('AIRTABLE_API_KEY:', process.env.AIRTABLE_API_KEY ? 'SET' : 'NOT SET');

  const config = {
    baseId: process.env.AIRTABLE_BASE_ID || 'YOUR_BASE_ID_HERE',
    apiKey: process.env.AIRTABLE_API_KEY || 'YOUR_API_KEY_HERE',
    tableName: 'RSVPs',
    get url() {
      return `https://api.airtable.com/v0/${this.baseId}/${this.tableName}`;
    }
  };

  console.log('Final config:', {
    baseId: config.baseId,
    apiKey: config.apiKey.substring(0, 10) + '...',
    tableName: config.tableName
  });

  res.setHeader('Content-Type', 'application/javascript');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.status(200).send(`window.AIRTABLE_CONFIG = ${JSON.stringify(config, null, 2)};`);
};
