# The Fold Studio Grand Opening - Landing Page

An immersive event landing page with a retro-future 70s terminal computer aesthetic, built with p5.js and featuring ASCII-style rendering, CRT post-processing effects, and mouse-reactive interactions.

## Features

- **10-second cinematic intro sequence** with overwhelming data-stream energy
- **ASCII-style text rendering** with monospaced terminal aesthetics
- **Large ASCII art title** ("THE FOLD") with random type-on/type-off animation
- **Dense background code typing** with verbose C++/CUDA code snippets (2000+ chars/second)
- **Separate buffer layer system** for clean main text rendering
- **Event info with black text on red background** styling
- **Mouse interaction** with text scramble effects
- **Flashing RSVP button** with color inversion animation
- **RSVP form** with Airtable integration
- **Responsive design** for desktop, tablet, and mobile
- **Parameterized colors** for easy customization
- **FPS monitoring** for performance optimization

## Quick Start

### Local Development
1. **Clone the repository:**
   ```bash
   git clone https://github.com/dcheesman/fold-studio-invite.git
   cd fold-studio-invite
   ```

2. **Set up configuration:**
   ```bash
   cp config.example.js config.js
   # Edit config.js with your Airtable credentials
   ```

3. **Open `index.html`** in a web browser

### Production Deployment (Vercel)
1. **Fork or clone** this repository
2. **Connect to Vercel:**
   - Go to [vercel.com](https://vercel.com)
   - Import your GitHub repository
   - Set framework to "Other" or "Static Site"
3. **Add environment variables:**
   - `AIRTABLE_BASE_ID`: Your Airtable base ID
   - `AIRTABLE_API_KEY`: Your Airtable API key
4. **Deploy!** - Vercel will auto-deploy on every push to `main`

### Airtable Setup
- Create an Airtable base with a table named "RSVPs"
- Add fields: Name (single line text), "Bringing Plus One" (checkbox), "Date/Time" (created time)
- Generate an API key from https://airtable.com/create/tokens
- Add credentials to `config.js` (local) or Vercel environment variables (production)

## File Structure

```
├── index.html              # Main HTML file with p5.js setup
├── sketch.js               # p5.js canvas implementation
├── config.js               # Configuration file (gitignored)
├── config.example.js       # Configuration template
├── favicon_io/             # Favicon files and web manifest
│   ├── favicon.ico         # Main favicon
│   ├── favicon-16x16.png   # 16x16 favicon
│   ├── favicon-32x32.png   # 32x32 favicon
│   ├── apple-touch-icon.png # Apple touch icon
│   ├── android-chrome-192x192.png # Android icon
│   ├── android-chrome-512x512.png # Android icon
│   └── site.webmanifest    # Web app manifest
├── .gitignore              # Git ignore rules
├── .cursorrules            # Project specifications and guidelines
├── head.txt                # ASCII art for head image
├── the_fold.txt            # ASCII art for "THE FOLD" title
└── README.md               # This file
```

## Deployment

### Automatic Deployment
- **GitHub:** `https://github.com/dcheesman/fold-studio-invite`
- **Vercel:** Auto-deploys on every push to `main` branch
- **Live URL:** Check Vercel dashboard for deployment URL

### Manual Deployment
1. Make changes locally
2. Test thoroughly
3. Commit and push to `main`:
   ```bash
   git add .
   git commit -m "feat: Description of changes"
   git push origin main
   ```
4. Vercel automatically deploys within 1-2 minutes

### Environment Variables (Production)
Set these in your Vercel dashboard:
- `AIRTABLE_BASE_ID`: `appCqw66jSnW2SnDC`
- `AIRTABLE_API_KEY`: Your Airtable personal access token

**Note:** The build process automatically injects both `AIRTABLE_BASE_ID` and `AIRTABLE_API_KEY` from environment variables into `config.js` during deployment, so the RSVP form will work with real Airtable integration in production.

## Configuration

### Colors
All colors are easily configurable in the `CONFIG` object:

```javascript
const CONFIG = {
    colors: {
        background: '#000000',    // Pure black background
        nearBlack: '#1a1a1a',     // Near-black for subtle elements
        grey: '#4a4a4a',          // Background text color
        lightGrey: '#6a6a6a',     // Lighter background text
        red: '#ff0033',           // Hero text color
        pureRed: '#ff0000',       // Pure red for main text
        darkRed: '#cc0000',       // Darker red variant
        gold: '#ffaa00',          // RSVP button color
        lightGold: '#ff8800',     // Lighter gold variant
        white: '#ffffff'          // Cursor color
    }
};
```

### Timing
Adjust the intro sequence timing:

```javascript
timing: {
    introDuration: 12000, // Total intro duration (12 seconds)
    scrollPhase: 5000,    // Background typing phase (5 seconds)
    typingPhase: 2000,    // Title typing phase (2 seconds)
    infoPhase: 4000,      // Event info phase (4 seconds)
    rsvpPhase: 1000       // RSVP phase (1 second)
}
```

### Text Content
Modify the displayed text and background phrases:

```javascript
text: {
    title: "THE FOLD",
    subtitle: "YOU ARE INVITED",
    date: "To a celebration",
    address: "8 years of The Fold",
    description: "at their new office",
    location: "located at 40w 100n Provo",
    time: "6:30pm - 9:30pm on the 23rd of October",
    refreshments: "drinks and refreshments provided",
    rsvpRequest: "please rsvp",
    closing: "we're looking forward to your initiation",
    rsvpText: "→ RSVP"
}
```

## Airtable Setup

1. **Create a new Airtable base**
2. **Create a table named "RSVPs"**
3. **Add the following fields:**
   - `Name` (Single line text)
   - `PlusOne` (Checkbox)
   - `Timestamp` (Created time - auto-generated)
4. **Generate an API key:**
   - Go to https://airtable.com/create/tokens
   - Create a new token with access to your base
   - Copy the token
5. **Update the configuration:**
   - Open `index.html`
   - Find the `AIRTABLE_CONFIG` object
   - Replace `YOUR_BASE_ID_HERE` with your base ID
   - Replace `YOUR_API_KEY_HERE` with your API key

## Architecture

The project uses a layered approach with p5.js:

1. **Background Layer:** Dense character grid with C++/CUDA code typing animation (2000+ chars/second)
2. **ASCII Art Buffer:** Separate layer for "THE FOLD" ASCII art with random type-on/type-off animation
3. **Main Text Buffer:** Separate p5.Graphics layer for event info with black text on red background
4. **HTML Layer:** Clickable RSVP element with flashing animation positioned on the character grid
5. **UI Layer:** White triangle cursor, FPS monitor, and interactive elements

**Performance Optimizations:**
- Post-processing effects disabled for better FPS
- Separate buffer system prevents text overwriting
- Random character typing (10-30 chars per frame) for fast background animation
- Optimized character grid rendering with proper push/pop state management
- FPS monitoring and adaptive frame rates

## Browser Compatibility

- **Chrome/Edge:** Full support
- **Firefox:** Full support
- **Safari:** Full support
- **Mobile browsers:** Optimized for mobile with reduced effects

## Performance Notes

- **Desktop:** 30fps target with all effects enabled
- **Mobile:** 24fps target with simplified effects for battery conservation
- **Post-processing:** Automatically disabled on mobile for better performance

## Customization

### Adding New Background Phrases
Edit the `backgroundText` array in the `CONFIG` object:

```javascript
backgroundText: [
    "Your custom phrase here",
    "Another cryptic message",
    // ... add more phrases
]
```

### Modifying Post-Processing Effects
Toggle effects in the sketch:

```javascript
let enableBloom = true;      // Glow effect on red/gold text
let enableScanlines = true;  // CRT scanline effect
let enableBlur = true;       // Soft blur effect
```

### Adjusting Mouse Interaction
Modify the scramble effect:

```javascript
let mouseProximity = 80;     // Radius in pixels
let scrambleTimeout = 300;   // Revert time in milliseconds
```

## Troubleshooting

### Canvas Not Loading
- Ensure you're serving the files from a web server (not opening directly)
- Check browser console for JavaScript errors
- Verify p5.js CDN is accessible

### Airtable Not Working
- Check that your API key has the correct permissions
- Verify the base ID is correct
- Ensure the table name is exactly "RSVPs"
- Check browser network tab for API errors

### Performance Issues
- Reduce `targetFrameRate` for slower devices
- Disable post-processing effects on mobile
- Check browser hardware acceleration settings

## License

This project is created for The Fold Studio Grand Opening event.

## Credits

Built with [p5.js](https://p5js.org/) and inspired by 1970s terminal aesthetics.

