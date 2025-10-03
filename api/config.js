// Serverless function to serve config with environment variables
export default function handler(req, res) {
  const config = {
    baseId: process.env.AIRTABLE_BASE_ID || 'YOUR_BASE_ID_HERE',
    apiKey: process.env.AIRTABLE_API_KEY || 'YOUR_API_KEY_HERE',
    tableName: 'RSVPs',
    get url() {
      return `https://api.airtable.com/v0/${this.baseId}/${this.tableName}`;
    }
  };

  res.setHeader('Content-Type', 'application/javascript');
  res.status(200).send(`window.AIRTABLE_CONFIG = ${JSON.stringify(config, null, 2)};`);
}
