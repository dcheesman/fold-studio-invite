// The Fold Studio Grand Opening - p5.js Sketch
// Retro-future 70s terminal computer aesthetic

function sketch(p) {
    // Configuration
    const CONFIG = {
        colors: {
            background: '#000000',
            nearBlack: '#1a1a1a',
            grey: '#4a4a4a',
            lightGrey: '#6a6a6a',
            red: '#ff0033',
            darkRed: '#cc0000',
            pureRed: '#ff0000',  // Pure red for main text
            gold: '#ffaa00',
            lightGold: '#ff8800'
        },
        timing: {
            introDuration: 8000, // 8 seconds total (5s background + 3s title/info)
            scrollPhase: 5000,   // 0-5s: background scroll
            typingPhase: 2000,   // 5-7s: title typing
            infoPhase: 1000      // 7-8s: event info
        },
        text: {
            title: "THE FOLD",
            subtitle: "Studio Grand Opening",
            date: "October 24, 2025",
            address: "[Address TBD]",
            description: "A night for the true believers.",
            rsvpText: "→ RSVP"
        },
        backgroundText: [
            "// Initialize neural network parameters",
            "constexpr auto MAX_ITERATIONS = 1000000;",
            "std::vector<std::unique_ptr<Layer>> network_layers;",
            "// Configure backpropagation algorithm",
            "auto learning_rate = 0.001f;",
            "auto momentum = 0.9f;",
            "// Memory allocation for tensor operations",
            "cudaMalloc(&d_weights, sizeof(float) * weight_count);",
            "// Error handling and validation",
            "if (error_code != CUDA_SUCCESS) {",
            "    throw std::runtime_error(\"CUDA allocation failed\");",
            "}",
            "// Matrix multiplication kernel",
            "__global__ void matmul_kernel(float* A, float* B, float* C) {",
            "    int idx = blockIdx.x * blockDim.x + threadIdx.x;",
            "    if (idx < N) {",
            "        C[idx] = A[idx] * B[idx];",
            "    }",
            "}",
            "// Synchronize device memory",
            "cudaDeviceSynchronize();",
            "// Check for memory leaks",
            "assert(allocated_memory == 0);",
            "// Performance profiling",
            "auto start_time = std::chrono::high_resolution_clock::now();",
            "// Process input data pipeline",
            "for (auto& batch : training_data) {",
            "    forward_pass(batch);",
            "    compute_loss(batch);",
            "    backward_pass(batch);",
            "    update_weights();",
            "}",
            "// Cleanup resources",
            "cudaFree(d_weights);",
            "// Final validation",
            "validate_model_accuracy();",
            "// Additional dense code snippets",
            "template<typename T> class Tensor {",
            "    std::vector<T> data;",
            "    std::vector<size_t> shape;",
            "public:",
            "    Tensor(const std::vector<size_t>& dims) : shape(dims) {",
            "        size_t total = 1;",
            "        for (auto dim : dims) total *= dim;",
            "        data.resize(total);",
            "    }",
            "};",
            "// GPU memory management",
            "cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);",
            "// Kernel launch configuration",
            "dim3 blockSize(256);",
            "dim3 gridSize((N + blockSize.x - 1) / blockSize.x);",
            "matmul_kernel<<<gridSize, blockSize>>>(A, B, C);",
            "// Error checking",
            "cudaError_t err = cudaGetLastError();",
            "if (err != cudaSuccess) {",
            "    fprintf(stderr, \"Kernel launch failed: %s\\n\", cudaGetErrorString(err));",
            "    exit(1);",
            "}",
            "// Memory optimization",
            "cudaMallocManaged(&unified_memory, size);",
            "cudaStream_t stream;",
            "cudaStreamCreate(&stream);",
            "// Asynchronous operations",
            "cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);",
            "cudaStreamSynchronize(stream);",
            "// Cleanup",
            "cudaStreamDestroy(stream);",
            "cudaFree(unified_memory);"
        ]
    };

    // Animation state
    let introPhase = 0; // 0: scroll, 1: typing, 2: color transition, 3: settled
    let phaseStartTime;
    let currentTime = 0;
    let isIntroComplete = false;
    
    // Grid system
    let charWidth, charHeight;
    let cols, rows;
    let fontSize;
    
        // Character grid system
        let charGrid = [];
        let gridCols, gridRows;
        let backgroundLastTypingTime = 0;
        let backgroundTypingSpeed = 5; // ms per character (200 chars/second)
        let currentTypingPosition = 0;
        let totalTypingPositions = 0;
        let typingQueue = [];
    
    // Typing animation
    let titleTypingQueue = [];
    let infoTypingQueue = [];
    let currentTypingIndex = 0;
    let titleLastTypingTime = 0;
    let infoLastTypingTime = 0;
    let typingSpeed = 5; // ms per character (much faster)
    let titleGrid = [];
    let infoGrid = [];
    
    // Color transitions
    let titleColor = CONFIG.colors.grey;
    let infoColor = CONFIG.colors.grey;
    let rsvpColor = CONFIG.colors.gold; // RSVP stays gold
    
    // Mouse interaction
    let mouseProximity = 80; // pixels
    let scrambledChars = {};
    let scrambleTimeouts = {};
    
    // Post-processing
    let scanlineOffset = 0;
    let enableBloom = true;
    let enableScanlines = true;
    let enableBlur = true;
    
    // Cursor
    let cursorSize = 12;
    
    // Responsive settings
    let targetFrameRate = 30;
    let isMobile = false;

    p.setup = function() {
        // Responsive setup
        isMobile = p.windowWidth < 768;
        if (isMobile) {
            targetFrameRate = 24;
            enableBloom = false;
        } else {
            targetFrameRate = 30;
            enableBloom = true;
        }
        
        p.createCanvas(p.windowWidth, p.windowHeight);
        p.pixelDensity(1); // Better performance on retina displays
        p.frameRate(targetFrameRate);
        
        // Calculate grid system
        fontSize = isMobile ? 16 : 18;
        charWidth = fontSize * 0.6; // Monospace character width
        charHeight = fontSize * 1.2; // Line height
        cols = p.floor(p.width / charWidth);
        rows = p.floor(p.height / charHeight);
        
        // Initialize character grid
        initializeCharGrid();
        
        // Start intro sequence
        phaseStartTime = p.millis();
        p.textFont('Courier New', fontSize);
        p.textAlign(p.LEFT, p.TOP);
    };

    p.draw = function() {
        currentTime = p.millis();
        let elapsed = currentTime - phaseStartTime;
        
        // Clear background
        p.background(CONFIG.colors.background);
        
        // Draw character grid
        drawCharGrid();
        
        // Handle intro sequence phases
        if (!isIntroComplete) {
            handleIntroSequence(elapsed);
        } else {
            // Post-intro settled state
            drawSettledContent();
        }
        
        // Draw cursor
        drawCursor();
        
        // Apply post-processing effects
        applyPostProcessing();
        
        // Update scroll offset for next frame
        scanlineOffset += 0.5;
    };

    function initializeCharGrid() {
        // Initialize character grid
        gridCols = cols;
        gridRows = rows;
        charGrid = [];
        
        // Initialize empty grid
        for (let y = 0; y < gridRows; y++) {
            charGrid[y] = [];
            for (let x = 0; x < gridCols; x++) {
                charGrid[y][x] = {
                    char: ' ',
                    color: CONFIG.colors.grey,
                    isTyped: false,
                    scrambleChar: null,
                    scrambleTimeout: null
                };
            }
        }
        
        // Initialize typing queue with background text
        initializeTypingQueue();
    }

    function initializeTypingQueue() {
        typingQueue = [];
        currentTypingPosition = 0;
        
        // Create a dense, code-like layout with much more content
        let selectedPhrases = CONFIG.backgroundText; // Use all phrases
        let positions = [];
        
        // Create many more positions for code-like text placement - fill entire screen
        for (let i = 0; i < selectedPhrases.length * 15; i++) { // 15x more code to fill screen
            let phrase = selectedPhrases[i % selectedPhrases.length];
            let x = p.floor(p.random(0, gridCols - phrase.length)); // Allow full width usage
            let y = p.floor(p.random(0, gridRows)); // Use entire height
            positions.push({x: x, y: y, text: phrase});
        }
        
        // Add more random single characters to fill gaps
        for (let i = 0; i < gridCols * gridRows / 10; i++) {
            let x = p.floor(p.random(0, gridCols));
            let y = p.floor(p.random(0, gridRows));
            let randomChars = ['#', '@', '$', '%', '&', '*', '+', '=', '~', '^', '|', '\\', '/', '-', '_'];
            let char = randomChars[p.floor(p.random(randomChars.length))];
            positions.push({x: x, y: y, text: char});
        }
        
        // Don't sort - keep random order for better distribution
        // positions.sort((a, b) => a.y - b.y || a.x - b.x);
        
        // Add to typing queue
        for (let pos of positions) {
            for (let i = 0; i < pos.text.length; i++) {
                if (pos.x + i < gridCols) {
                    typingQueue.push({
                        x: pos.x + i,
                        y: pos.y,
                        char: pos.text[i],
                        color: CONFIG.colors.grey
                    });
                }
            }
        }
        
        totalTypingPositions = typingQueue.length;
    }

    function drawCharGrid() {
        p.push();
        p.textAlign(p.LEFT, p.TOP);
        
        // Update typing animation
        updateCharGridTyping();
        
        // Draw the character grid
        for (let y = 0; y < gridRows; y++) {
            for (let x = 0; x < gridCols; x++) {
                let cell = charGrid[y][x];
                let screenX = x * charWidth;
                let screenY = y * charHeight;
                
                // Only draw if cell is visible and has content
                if (screenY > -charHeight && screenY < p.height && cell.char !== ' ') {
                    let displayChar = cell.char;
                    let displayColor = cell.color;
                    
                    // Apply mouse scramble effect only to background code, not main text
                    if (isIntroComplete && !cell.isMainText) {
                        let scrambled = applyMouseScrambleToCell(x, y, screenX, screenY);
                        if (scrambled) {
                            displayChar = scrambled;
                        }
                    }
                    
                    p.fill(displayColor);
                    p.text(displayChar, screenX, screenY);
                }
            }
        }
        p.pop();
    }

    function updateCharGridTyping() {
        if (currentTime - backgroundLastTypingTime < backgroundTypingSpeed) return;
        
        // Type next character from queue
        if (currentTypingPosition < typingQueue.length) {
            let nextChar = typingQueue[currentTypingPosition];
            let x = nextChar.x;
            let y = nextChar.y;
            
            if (x < gridCols && y < gridRows && !charGrid[y][x].isMainText) {
                charGrid[y][x].char = nextChar.char;
                charGrid[y][x].color = nextChar.color;
                charGrid[y][x].isTyped = true;
            }
            
            currentTypingPosition++;
            backgroundLastTypingTime = currentTime;
        }
    }

    function applyMouseScrambleToCell(x, y, screenX, screenY) {
        let mouseX = p.mouseX;
        let mouseY = p.mouseY;
        let distance = p.dist(screenX, screenY, mouseX, mouseY);
        
        if (distance < mouseProximity) {
            let cell = charGrid[y][x];
            let cellKey = `${x}-${y}`;
            
            if (!cell.scrambleChar) {
                cell.scrambleChar = randomScrambleChar();
                
                // Set timeout to revert character
                if (cell.scrambleTimeout) {
                    clearTimeout(cell.scrambleTimeout);
                }
                cell.scrambleTimeout = setTimeout(() => {
                    cell.scrambleChar = null;
                    cell.scrambleTimeout = null;
                }, 300);
            }
            return cell.scrambleChar;
        }
        return null;
    }

    function handleIntroSequence(elapsed) {
        if (introPhase === 0 && elapsed > 5000) { // Wait 5 seconds before showing title
            // Start typing phase
            introPhase = 1;
            phaseStartTime = currentTime;
            initializeTitleTyping();
        } else if (introPhase === 1 && elapsed > 2000) { // 2 seconds of title typing
            // Start info typing phase
            introPhase = 2;
            phaseStartTime = currentTime;
            initializeInfoTyping();
        } else if (introPhase === 2 && elapsed > 2000) { // 2 seconds of info typing
            // Start color transition phase
            introPhase = 3;
            phaseStartTime = currentTime;
        } else if (introPhase === 3 && elapsed > 1000) { // 1 second of color transition
            // Intro complete
            introPhase = 4;
            isIntroComplete = true;
            phaseStartTime = currentTime;
        }
        
        // Handle typing animation
        if (introPhase === 1) {
            handleTitleTyping();
        } else if (introPhase === 2) {
            handleTitleTyping();
            handleInfoTyping();
        } else if (introPhase >= 3) {
            handleTitleTyping();
            handleInfoTyping();
        }
        
        // No color transitions needed - typing directly as red
        
        // Draw content based on phase
        drawIntroContent();
    }

    function initializeTitleTyping() {
        titleGrid = [];
        titleTypingQueue = [];
        
        // Create grid positions for title
        let titleText = CONFIG.text.title;
        let startX = p.floor(gridCols * 0.2); // Left aligned
        let startY = p.floor(gridRows * 0.4); // Center vertically
        
        // Initialize grid
        for (let i = 0; i < titleText.length; i++) {
            let x = startX + i;
            let y = startY;
            if (x < gridCols && y < gridRows) {
                titleGrid.push({x: x, y: y, char: titleText[i], typed: false});
            }
        }
        
        // Shuffle the typing order
        titleTypingQueue = [...titleGrid];
        for (let i = titleTypingQueue.length - 1; i > 0; i--) {
            let j = p.floor(p.random(i + 1));
            [titleTypingQueue[i], titleTypingQueue[j]] = [titleTypingQueue[j], titleTypingQueue[i]];
        }
        
        currentTypingIndex = 0;
    }

    function initializeInfoTyping() {
        infoGrid = [];
        infoTypingQueue = [];
        
        // Create grid positions for info text
        let infoTexts = [
            CONFIG.text.subtitle,
            CONFIG.text.date,
            CONFIG.text.address,
            CONFIG.text.description
        ];
        
        let startX = p.floor(gridCols * 0.2); // Left aligned
        let startY = p.floor(gridRows * 0.55); // Below title
        
        // Initialize grid for each info line
        for (let lineIndex = 0; lineIndex < infoTexts.length; lineIndex++) {
            let text = infoTexts[lineIndex];
            for (let i = 0; i < text.length; i++) {
                let x = startX + i;
                let y = startY + lineIndex * 2; // 2 lines spacing
                if (x < gridCols && y < gridRows) {
                    infoGrid.push({x: x, y: y, char: text[i], typed: false, lineIndex: lineIndex});
                }
            }
        }
        
        // Shuffle the typing order
        infoTypingQueue = [...infoGrid];
        for (let i = infoTypingQueue.length - 1; i > 0; i--) {
            let j = p.floor(p.random(i + 1));
            [infoTypingQueue[i], infoTypingQueue[j]] = [infoTypingQueue[j], infoTypingQueue[i]];
        }
        
        console.log('Info typing queue initialized with', infoTypingQueue.length, 'characters');
    }

    function handleTitleTyping() {
        if (currentTime - titleLastTypingTime > typingSpeed && currentTypingIndex < titleTypingQueue.length) {
            let charData = titleTypingQueue[currentTypingIndex];
            if (charData.x < gridCols && charData.y < gridRows) {
                charGrid[charData.y][charData.x].char = charData.char;
                charGrid[charData.y][charData.x].color = CONFIG.colors.pureRed;
                charGrid[charData.y][charData.x].isTyped = true;
                charGrid[charData.y][charData.x].isMainText = true;
                charData.typed = true;
            }
            currentTypingIndex++;
            titleLastTypingTime = currentTime;
        }
    }

    function handleInfoTyping() {
        if (currentTime - infoLastTypingTime > typingSpeed) {
            // Find next untyped character in info queue
            let found = false;
            for (let i = 0; i < infoTypingQueue.length && !found; i++) {
                let charData = infoTypingQueue[i];
                if (!charData.typed && charData.x < gridCols && charData.y < gridRows) {
                    // Always place info text, even if it overwrites background
                    charGrid[charData.y][charData.x].char = charData.char;
                    charGrid[charData.y][charData.x].color = CONFIG.colors.pureRed;
                    charGrid[charData.y][charData.x].isTyped = true;
                    charGrid[charData.y][charData.x].isMainText = true;
                    charData.typed = true;
                    found = true;
                }
            }
            infoLastTypingTime = currentTime;
        }
    }

    function handleColorTransitions(elapsed) {
        let transitionProgress = p.constrain(elapsed / 1000, 0, 1); // 1 second transition
        
        // Easing function for smooth color transition
        let eased = easeInOutCubic(transitionProgress);
        
        // Transition title from grey to pure red
        titleColor = p.lerpColor(
            p.color(CONFIG.colors.grey),
            p.color(CONFIG.colors.pureRed),
            eased
        );
        
        // Transition info text from grey to pure red
        infoColor = p.lerpColor(
            p.color(CONFIG.colors.grey),
            p.color(CONFIG.colors.pureRed),
            eased
        );
    }

    function drawIntroContent() {
        // Title and info are now drawn as part of the character grid
        // Only need to draw RSVP text separately
        if (introPhase >= 3) {
            drawRSVPText();
        }
    }

    function drawRSVPText() {
        // Add RSVP text to the character grid
        let rsvpText = CONFIG.text.rsvpText;
        let startX = p.floor(gridCols * 0.2);
        let startY = p.floor(gridRows * 0.7);
        
        for (let i = 0; i < rsvpText.length; i++) {
            let x = startX + i;
            let y = startY;
            if (x < gridCols && y < gridRows) {
                charGrid[y][x].char = rsvpText[i];
                charGrid[y][x].color = rsvpColor;
                charGrid[y][x].isTyped = true;
                charGrid[y][x].isMainText = true; // Mark as main text
            }
        }
    }

    function drawSettledContent() {
        // All content is now drawn as part of the character grid
        // No additional drawing needed
    }

    function drawCursor() {
        p.push();
        p.fill(CONFIG.colors.gold);
        p.noStroke();
        
        // Draw angled triangle pointer
        let cursorX = p.mouseX;
        let cursorY = p.mouseY;
        
        // Create triangle pointing right
        p.triangle(
            cursorX, cursorY - cursorSize/2,  // Top point
            cursorX, cursorY + cursorSize/2,  // Bottom point  
            cursorX + cursorSize, cursorY     // Right point
        );
        
        p.pop();
    }


    function randomScrambleChar() {
        let scrambleChars = ['#', '@', '$', '%', '&', '*', '+', '=', '~', '^'];
        return scrambleChars[p.floor(p.random(scrambleChars.length))];
    }

    // Utility functions for future ASCII art generation
    function setCharAt(x, y, char, color = CONFIG.colors.grey) {
        if (x >= 0 && x < gridCols && y >= 0 && y < gridRows) {
            charGrid[y][x].char = char;
            charGrid[y][x].color = color;
            charGrid[y][x].isTyped = true;
        }
    }

    function getCharAt(x, y) {
        if (x >= 0 && x < gridCols && y >= 0 && y < gridRows) {
            return charGrid[y][x];
        }
        return null;
    }

    function clearGrid() {
        for (let y = 0; y < gridRows; y++) {
            for (let x = 0; x < gridCols; x++) {
                charGrid[y][x].char = ' ';
                charGrid[y][x].isTyped = false;
            }
        }
    }

    function addTextToGrid(text, startX, startY, color = CONFIG.colors.grey) {
        for (let i = 0; i < text.length; i++) {
            if (startX + i < gridCols && startY < gridRows) {
                setCharAt(startX + i, startY, text[i], color);
            }
        }
    }

    // Future function for ASCII art from image/video
    function generateASCIIFromImage(img, targetWidth = null, targetHeight = null) {
        if (!img) return;
        
        let w = targetWidth || gridCols;
        let h = targetHeight || gridRows;
        
        // ASCII character set from dark to light
        let asciiChars = [' ', '.', ':', ';', 'o', 'x', '%', '#', '@'];
        
        // Sample image and convert to ASCII
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                let imgX = p.map(x, 0, w, 0, img.width);
                let imgY = p.map(y, 0, h, 0, img.height);
                
                let pixel = img.get(imgX, imgY);
                let brightness = (pixel[0] + pixel[1] + pixel[2]) / 3;
                let charIndex = p.floor(p.map(brightness, 0, 255, 0, asciiChars.length - 1));
                
                setCharAt(x, y, asciiChars[charIndex], CONFIG.colors.grey);
            }
        }
    }

    function applyPostProcessing() {
        if (!enableScanlines && !enableBloom && !enableBlur) return;
        
        p.loadPixels();
        
        // Apply scanlines
        if (enableScanlines) {
            applyScanlines();
        }
        
        // Apply bloom effect (simplified)
        if (enableBloom) {
            applyBloom();
        }
        
        // Apply blur
        if (enableBlur) {
            applyBlur();
        }
        
        p.updatePixels();
    }

    function applyScanlines() {
        for (let y = 0; y < p.height; y += 2) {
            if (y % 4 === 0) { // Every 4th pixel row
                for (let x = 0; x < p.width; x++) {
                    let index = (y * p.width + x) * 4;
                    p.pixels[index] *= 0.8;     // R
                    p.pixels[index + 1] *= 0.8; // G
                    p.pixels[index + 2] *= 0.8; // B
                }
            }
        }
    }

    function applyBloom() {
        // Simple bloom effect - brighten red and gold colors
        for (let i = 0; i < p.pixels.length; i += 4) {
            let r = p.pixels[i];
            let g = p.pixels[i + 1];
            let b = p.pixels[i + 2];
            
            // Check if pixel is red or gold-ish
            if ((r > 200 && g < 100 && b < 100) || (r > 200 && g > 150 && b < 100)) {
                p.pixels[i] = p.min(255, r * 1.1);     // R
                p.pixels[i + 1] = p.min(255, g * 1.05); // G
                p.pixels[i + 2] = p.min(255, b * 1.05); // B
            }
        }
    }

    function applyBlur() {
        // Simple box blur
        let tempPixels = p.pixels.slice();
        let blurRadius = 1;
        
        for (let y = blurRadius; y < p.height - blurRadius; y++) {
            for (let x = blurRadius; x < p.width - blurRadius; x++) {
                let r = 0, g = 0, b = 0, a = 0;
                let count = 0;
                
                for (let dy = -blurRadius; dy <= blurRadius; dy++) {
                    for (let dx = -blurRadius; dx <= blurRadius; dx++) {
                        let index = ((y + dy) * p.width + (x + dx)) * 4;
                        r += tempPixels[index];
                        g += tempPixels[index + 1];
                        b += tempPixels[index + 2];
                        a += tempPixels[index + 3];
                        count++;
                    }
                }
                
                let index = (y * p.width + x) * 4;
                p.pixels[index] = r / count;
                p.pixels[index + 1] = g / count;
                p.pixels[index + 2] = b / count;
                p.pixels[index + 3] = a / count;
            }
        }
    }

    function easeInOutCubic(t) {
        return t < 0.5 ? 4 * t * t * t : 1 - p.pow(-2 * t + 2, 3) / 2;
    }

    // Mouse interaction
    p.mousePressed = function() {
        if (isIntroComplete) {
            // Check if mouse clicked on RSVP text (now on grid)
            let rsvpText = CONFIG.text.rsvpText;
            let startX = p.floor(gridCols * 0.2);
            let startY = p.floor(gridRows * 0.7);
            
            // Convert grid coordinates to screen coordinates
            let screenX = startX * charWidth;
            let screenY = startY * charHeight;
            let textWidth = rsvpText.length * charWidth;
            let textHeight = charHeight;
            
            if (p.mouseX > screenX && 
                p.mouseX < screenX + textWidth &&
                p.mouseY > screenY && 
                p.mouseY < screenY + textHeight) {
                
                // Show RSVP form
                if (typeof showRSVPForm === 'function') {
                    showRSVPForm();
                }
            }
        }
    };

    // Window resize handling
    p.windowResized = function() {
        p.resizeCanvas(p.windowWidth, p.height);
        
        // Recalculate grid
        isMobile = p.windowWidth < 768;
        fontSize = isMobile ? 16 : 18;
        charWidth = fontSize * 0.6;
        charHeight = fontSize * 1.2;
        cols = p.floor(p.width / charWidth);
        rows = p.floor(p.height / charHeight);
        
        // Reinitialize character grid
        initializeCharGrid();
        
        // Update frame rate
        if (isMobile) {
            targetFrameRate = 24;
            enableBloom = false;
        } else {
            targetFrameRate = 30;
            enableBloom = true;
        }
        p.frameRate(targetFrameRate);
    };

    // Prevent context menu on right click
    p.mousePressed = function() {
        if (p.mouseButton === p.RIGHT) {
            return false;
        }
        
        // Handle RSVP click
        if (isIntroComplete) {
            let rsvpY = p.height * 0.55 + fontSize * 1.5 * 4.5;
            let rsvpTextWidth = p.textWidth(CONFIG.text.rsvpText);
            let rsvpX = p.width / 2;
            
            if (p.mouseX > rsvpX - rsvpTextWidth / 2 && 
                p.mouseX < rsvpX + rsvpTextWidth / 2 &&
                p.mouseY > rsvpY - fontSize / 2 && 
                p.mouseY < rsvpY + fontSize / 2) {
                
                if (typeof showRSVPForm === 'function') {
                    showRSVPForm();
                }
            }
        }
    };
}

