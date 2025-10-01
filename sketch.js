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
            introDuration: 10000, // 10 seconds total
            scrollPhase: 5000,    // 0-5s: background typing
            typingPhase: 2000,    // 5-7s: title typing
            infoPhase: 2000,      // 7-9s: event info
            rsvpPhase: 1000       // 9-10s: RSVP
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
    let backgroundTypingSpeed = 0.3; // ms per character (100,000 chars/second with multi-char per frame)
    let currentTypingPosition = 0;
    let totalTypingPositions = 0;
    let typingQueue = [];
    
    // Main text buffer (separate layer)
    let mainTextBuffer;
    
    // HTML RSVP element
    let rsvpElement;
    
    // Typing animation - simplified system
    let titleText = "";
    let infoText = "";
    let titleTypingIndex = 0;
    let infoTypingIndex = 0;
    let titleLastTypingTime = 0;
    let infoLastTypingTime = 0;
    let typingSpeed = 5; // ms per character (much faster)
    
    // Text positions
    let titleX, titleY;
    let infoLines = [];
    let infoStartY;
    
    // Color transitions (removed - text types directly as red/gold)
    
    // Mouse interaction
    let mouseProximity = 80; // pixels
    let scrambledChars = {};
    let scrambleTimeouts = {};
    
    // Post-processing (disabled for performance)
    // let scanlineOffset = 0;
    // let enableBloom = false;
    // let enableScanlines = false;
    // let enableBlur = false;
    
    // Cursor
    let cursorSize = 12;
    
    // FPS monitoring
    let fps = 0;
    let frameCount = 0;
    let lastFpsTime = 0;
    
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
        
        // Initialize main text buffer
        mainTextBuffer = p.createGraphics(p.width, p.height);
        mainTextBuffer.clear(); // Transparent background
        
        // Initialize HTML RSVP element
        rsvpElement = document.getElementById('rsvp-clickable');
        if (rsvpElement) {
            rsvpElement.addEventListener('click', function() {
                document.getElementById('rsvp-overlay').classList.add('visible');
            });
        }
        
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
        
        // Draw character grid (background can write anywhere)
        drawCharGrid();
        
        // Clear main text buffer
        mainTextBuffer.clear();
        
        // Handle intro sequence phases and main text (on separate buffer)
        if (!isIntroComplete) {
            // Hide RSVP element during intro phases
            if (rsvpElement && introPhase < 3) {
                rsvpElement.style.display = 'none';
            }
            handleIntroSequence(elapsed);
        } else {
            // Post-intro settled state
            drawSettledContent();
        }
        
        // Draw main text buffer on top
        p.image(mainTextBuffer, 0, 0);
        
        // Draw cursor
        drawCursor();
        
        // Draw FPS monitor
        drawFPSMonitor();
        
        // Post-processing disabled for performance
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
        
        // Create moderate amount of code for performance
        for (let i = 0; i < selectedPhrases.length * 5; i++) { // Reduced from 15x to 5x
            let phrase = selectedPhrases[i % selectedPhrases.length];
            let x = p.floor(p.random(0, gridCols - phrase.length)); // Allow full width usage
            let y = p.floor(p.random(0, gridRows)); // Use entire height
            positions.push({x: x, y: y, text: phrase});
        }
        
        // Add fewer random single characters for performance
        for (let i = 0; i < gridCols * gridRows / 20; i++) { // Reduced from /10 to /20
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
                    if (isIntroComplete) {
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
        
        // Calculate how many characters to type this frame based on speed
        let charsToType = 1;
        if (backgroundTypingSpeed < 1) {
            // For very fast speeds, type multiple characters per frame
            charsToType = Math.floor(1 / backgroundTypingSpeed);
        }
        
        // Type multiple characters from queue
        for (let i = 0; i < charsToType && currentTypingPosition < typingQueue.length; i++) {
            let nextChar = typingQueue[currentTypingPosition];
            let x = nextChar.x;
            let y = nextChar.y;
            
            if (x < gridCols && y < gridRows) {
                charGrid[y][x].char = nextChar.char;
                charGrid[y][x].color = nextChar.color;
                charGrid[y][x].isTyped = true;
            }
            
            currentTypingPosition++;
        }
        
        backgroundLastTypingTime = currentTime;
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
            initializeSimpleTyping();
        } else if (introPhase === 1 && elapsed > 2000) { // 2 seconds of title typing
            // Start info typing phase
            introPhase = 2;
            phaseStartTime = currentTime;
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

    function initializeSimpleTyping() {
        // Set up simple text positions
        titleX = p.floor(gridCols * 0.2);
        titleY = p.floor(gridRows * 0.4);
        
        infoLines = [
            CONFIG.text.subtitle,
            CONFIG.text.date,
            CONFIG.text.address,
            CONFIG.text.description
        ];
        infoStartY = p.floor(gridRows * 0.55);
        
        // Reset typing indices
        titleTypingIndex = 0;
        infoTypingIndex = 0;
        titleText = "";
        infoText = "";
    }

    function handleTitleTyping() {
        if (currentTime - titleLastTypingTime > typingSpeed && titleTypingIndex < CONFIG.text.title.length) {
            titleText += CONFIG.text.title[titleTypingIndex];
            titleTypingIndex++;
            titleLastTypingTime = currentTime;
        }
    }

    function handleInfoTyping() {
        if (currentTime - infoLastTypingTime > typingSpeed) {
            // Build info text line by line
            let currentLine = Math.floor(infoTypingIndex / 50); // Approximate chars per line
            if (currentLine < infoLines.length) {
                let lineText = infoLines[currentLine];
                let lineIndex = infoTypingIndex % 50;
                if (lineIndex < lineText.length) {
                    infoText += lineText[lineIndex];
                } else if (lineIndex === lineText.length) {
                    infoText += "\n"; // Add newline after each line
                }
                infoTypingIndex++;
            }
            infoLastTypingTime = currentTime;
        }
    }

    // Color transitions removed - text types directly as red/gold

    function drawIntroContent() {
        // Draw title text
        if (introPhase >= 1 && titleText.length > 0) {
            drawSimpleText(titleText, titleX, titleY, CONFIG.colors.pureRed, 2);
        }
        
        // Draw info text
        if (introPhase >= 2 && infoText.length > 0) {
            drawSimpleText(infoText, titleX, infoStartY, CONFIG.colors.pureRed, 1);
        }
        
        // Draw RSVP text
        if (introPhase >= 3) {
            // Show and position HTML RSVP element instead of drawing to buffer
            if (rsvpElement) {
                let rsvpX = titleX * charWidth;
                let rsvpY = (infoStartY + 8) * charHeight;
                
                rsvpElement.style.left = rsvpX + 'px';
                rsvpElement.style.top = rsvpY + 'px';
                rsvpElement.style.fontSize = fontSize + 'px';
                rsvpElement.style.display = 'block';
            }
        }
    }
    
    function drawSimpleText(text, startX, startY, color, size) {
        mainTextBuffer.push();
        mainTextBuffer.fill(color);
        mainTextBuffer.textAlign(mainTextBuffer.LEFT, mainTextBuffer.TOP);
        mainTextBuffer.textSize(fontSize * size);
        mainTextBuffer.textFont('Courier New', fontSize * size);
        
        // Draw text with black background to overwrite
        let lines = text.split('\n');
        for (let i = 0; i < lines.length; i++) {
            let x = startX * charWidth;
            let y = (startY + i * 2) * charHeight;
            
            // Draw black background rectangle
            mainTextBuffer.fill(CONFIG.colors.background);
            mainTextBuffer.noStroke();
            mainTextBuffer.rect(x - 2, y - 2, lines[i].length * charWidth + 4, charHeight + 4);
            
            // Draw text
            mainTextBuffer.fill(color);
            mainTextBuffer.text(lines[i], x, y);
        }
        mainTextBuffer.pop();
    }


    function drawSettledContent() {
        // Draw all main text in settled state
        if (titleText.length > 0) {
            drawSimpleText(titleText, titleX, titleY, CONFIG.colors.pureRed, 2);
        }
        
        if (infoText.length > 0) {
            drawSimpleText(infoText, titleX, infoStartY, CONFIG.colors.pureRed, 1);
        }
        
        // Show and position HTML RSVP element
        if (rsvpElement) {
            let rsvpX = titleX * charWidth;
            let rsvpY = (infoStartY + 8) * charHeight;
            
            rsvpElement.style.left = rsvpX + 'px';
            rsvpElement.style.top = rsvpY + 'px';
            rsvpElement.style.fontSize = fontSize + 'px';
            rsvpElement.style.display = 'block';
        }
    }

    function drawCursor() {
        p.push();
        p.fill(CONFIG.colors.gold);
        p.noStroke();
        
        // Draw isosceles triangle pointer with sharpest point as active end
        let cursorX = p.mouseX;
        let cursorY = p.mouseY;
        let baseWidth = cursorSize * 0.6;  // Shorter base for sharper triangle
        let height = cursorSize;           // Full height
        
        // Create isosceles triangle pointing right with sharpest point
        p.triangle(
            cursorX, cursorY - baseWidth/2,     // Top left point
            cursorX, cursorY + baseWidth/2,     // Bottom left point  
            cursorX + height, cursorY           // Right point (sharpest)
        );
        
        p.pop();
    }

    function drawFPSMonitor() {
        // Update FPS calculation
        frameCount++;
        if (currentTime - lastFpsTime >= 1000) { // Update every second
            fps = frameCount;
            frameCount = 0;
            lastFpsTime = currentTime;
        }
        
        // Draw FPS in top left
        p.push();
        p.fill(CONFIG.colors.gold);
        p.textAlign(p.LEFT, p.TOP);
        p.textSize(12);
        p.text(`FPS: ${fps}`, 10, 10);
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

    // Post-processing functions removed (disabled for performance)

    function easeInOutCubic(t) {
        return t < 0.5 ? 4 * t * t * t : 1 - p.pow(-2 * t + 2, 3) / 2;
    }

    // Mouse interaction
    // Mouse click handling removed - RSVP is now HTML element

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
        
        // Recreate main text buffer with new dimensions
        mainTextBuffer = p.createGraphics(p.width, p.height);
        mainTextBuffer.clear(); // Transparent background
        
        // Update RSVP element font size
        if (rsvpElement) {
            rsvpElement.style.fontSize = fontSize + 'px';
        }
        
        // Update frame rate
        if (isMobile) {
            targetFrameRate = 24;
        } else {
            targetFrameRate = 30;
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

