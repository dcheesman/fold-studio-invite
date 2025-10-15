# Video Recording Guide

## Overview
The Fold invitation page now includes video recording capabilities for creating social media content. You can record high-quality 30-second videos in two formats:

- **Square (1080x1080)**: Perfect for Instagram posts
- **Vertical (1080x1920)**: Perfect for TikTok and Instagram Stories

## How to Record

### Option 1: Using the Recording Interface
1. Navigate to `/record.html` on your site
2. Click "Record Square" or "Record Vertical"
3. Wait 30 seconds for automatic recording
4. Video will automatically download as a `.webm` file

### Option 2: Using Browser Console
1. Go to your Friday page (`/friday.html`)
2. Open browser developer tools (F12)
3. In the console, type:
   - `startSquareRecording()` for square format
   - `startVerticalRecording()` for vertical format
4. Wait 30 seconds, then type `stopRecording()`

## Technical Details

### Video Specifications
- **Resolution**: 1080x1080 (square) or 1080x1920 (vertical)
- **Frame Rate**: 30fps
- **Duration**: 30 seconds
- **Format**: WebM (VP9 codec)
- **Quality**: High quality, no frame drops

### Browser Compatibility
- Chrome/Edge: Full support
- Firefox: Full support
- Safari: Limited support (may need to convert format)

## Converting to Other Formats

### For Instagram/TikTok Upload
1. Use online converter (e.g., CloudConvert) to convert WebM to MP4
2. Or use FFmpeg: `ffmpeg -i input.webm -c:v libx264 -c:a aac output.mp4`

### For Best Quality
- The WebM format provides excellent quality
- For social media, MP4 is more universally accepted
- Consider using H.264 codec for maximum compatibility

## Tips for Best Results

1. **Use Chrome or Edge** for best recording performance
2. **Close other tabs** to ensure maximum resources for recording
3. **Wait for full animation** before starting recording
4. **Test first** with a short recording to ensure everything works
5. **Check file size** - 30-second videos should be 5-15MB

## Troubleshooting

### Recording Not Starting
- Check browser console for errors
- Ensure you're using a supported browser
- Try refreshing the page

### Poor Quality
- Close other applications
- Use Chrome for best performance
- Check your system resources

### File Not Downloading
- Check browser download settings
- Look in your Downloads folder
- Try right-clicking the download link if it appears

## File Naming
Recorded files are automatically named:
- `fold-square-[timestamp].webm`
- `fold-vertical-[timestamp].webm`

This makes it easy to identify and organize your recordings.
