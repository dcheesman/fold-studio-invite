# The Fold Studio Grand Opening - Landing Page

An immersive event landing page with a retro-future 70s terminal computer aesthetic, built with p5.js and featuring ASCII-style rendering, CRT post-processing effects, and mouse-reactive interactions.

## Features

- **10-second cinematic intro sequence** with overwhelming data-stream energy
- **ASCII-style text rendering** with monospaced terminal aesthetics
- **Dense background code typing** with verbose C++/CUDA code snippets
- **Separate buffer layer system** for clean main text rendering
- **Mouse interaction** with text scramble effects
- **RSVP form** with Airtable integration
- **Responsive design** for desktop, tablet, and mobile
- **Parameterized colors** for easy customization
- **FPS monitoring** for performance optimization

## Quick Start

1. **Open `index.html`** in a web browser
2. **Configure Airtable** (optional):
   - Create an Airtable base with a table named "RSVPs"
   - Add fields: Name (single line text), PlusOne (checkbox), Timestamp (created time)
   - Generate an API key from https://airtable.com/create/tokens
   - Update the `AIRTABLE_CONFIG` object in `index.html` with your credentials

## File Structure

```
├── index.html          # Main HTML file with p5.js setup
├── sketch.js           # p5.js canvas implementation
├── config.js           # Configuration file (optional)
├── .cursorrules        # Project specifications and guidelines
└── README.md           # This file
```

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
        gold: '#ffaa00',          // Event info color
        lightGold: '#ff8800'      // Lighter gold variant
    }
};
```

### Timing
Adjust the intro sequence timing:

```javascript
timing: {
    introDuration: 10000, // Total intro duration (10 seconds)
    scrollPhase: 5000,    // Background typing phase (5 seconds)
    typingPhase: 2000,    // Title typing phase (2 seconds)
    infoPhase: 2000,      // Event info phase (2 seconds)
    rsvpPhase: 1000       // RSVP phase (1 second)
}
```

### Text Content
Modify the displayed text and background phrases:

```javascript
text: {
    title: "THE FOLD",
    subtitle: "Studio Grand Opening",
    date: "October 24, 2025",
    address: "[Address TBD]",
    description: "A night for the true believers.",
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

