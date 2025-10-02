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
        // Title ASCII art
        titleAsciiArt: [
            "                                    ░░░░░░░░  ░░░   ░░  ░░░░░░░░    ░░░  ░░░ ░░░░                                                  ",
            "                                     ░░▒▒░░   ░▒░   ░░  ░▒▒         ▒▒░ ░▓▒░ ░▒▒░                                                  ",
            "                                       ░░     ░▒▒░░░▒░  ░▒▒▒▒▒▒░    ▒▒░ ░▓▒░ ░▓▒░                                                  ",
            "                                       ░░     ░▒▒▒░▒▒░  ░▒▒▓▓▓▓░    ▒▒░ ░▓▒░ ░▓▒░                                                  ",
            "                                       ░░     ░▒░   ░░  ░▓▓▓▓▓▓░    ▒▒░ ░▓▒░ ░▓▒░                                                  ",
            "                                                                    ▒▒░ ░▓▒░ ░▓▒░                                                  ",
            "                                                                    ▒▒░ ░▓▒░ ░▓▒░                                                  ",
            "    ░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░    ░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░    ▒▒░ ░▓▒░ ░▓▒░                  ░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░    ",
            "    ░▒▒▒                            ░▒▒▒                    ▒▒▒░    ▒▒░ ░▓▒░ ░▓▒░                                          ▒▒▒░    ",
            "    ░▒▒▒ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░    ░▒▒▒ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ ░▒▒░    ▒▒░ ░▓▒░ ░▓▒░                  ░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ ▒▒▒░    ",
            "    ░▒▒░ ░▒▒▒░░░░░░░░░░░░░░░░░░░    ░▒▒░ ░▒▒░░░░░░░░░░▒▒▒▒░ ░▓▒░    ▒▒░ ░▓▒░ ░▓▒░                  ░░░░░░░░░░░░░░░░░░░▒▒▒░ ░▒▒░    ",
            "    ░▒▒░ ░▒▒▒                       ░▒▒░ ░▒░            ▓▒░ ░▓▒░    ▒▒░ ░▓▒░ ░▓▒░                                     ░▒▒░ ░▒▒░    ",
            "    ░▒▒░ ░▒▒▒░░░░░░░░░░░░░░░░░░░    ░▒▒░ ░▒░  ░░░░░░░░  ▓▒░ ░▓▒░    ▒▒░ ░▓▒░ ░▓▒░                  ░░░░░░░░░░░░░░░░░░ ░▒▒░ ░▒▒░    ",
            "    ░▒▒░ ░░░░░░░░░░░░░░░░░░░░░░░    ░▒▒░ ░▒░  ░▒▒▒▒▒▒░  ▓▒░ ░▓▒░    ▒▒░ ░▓▒░ ░▓▒░                  ░░░░░░░░░░░░░▒▒▒▒░ ░▒▒░ ░▒▒░    ",
            "    ░▒▒░                            ░▒▒░ ░▒░  ░▒▒  ▒▒░  ▓▒░ ░▓▒░    ▒▒░ ░▓▒░ ░▓▒░                                ▒▒▒░ ░▒▒░ ░▒▒░    ",
            "    ░▒▒░ ░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░    ░▒▒░ ░▒░  ░▒▒░▒▒▒░  ▓▒░ ░▓▒░    ▒▒░ ░▓▒░ ░▒▒▒▒░░░░░░░░░░░░░    ░░░░░░░░░░░░░░▒▒▓░ ░▒▒░ ░▒▒░    ",
            "    ░▒▒░ ░▒▒▒                       ░▒▒░ ░▒░            ▓▒░ ░▓▒░    ▒▒░ ░▓▒░                                          ░▒▒░ ░▒▒░    ",
            "    ░▒▒░ ░▒▒▒                       ░▒▒░ ░▒▒░░▒▒▒▒▒▒▒▒▒▒▒▒░ ░▓▒░    ▒▒░ ░▓▒▒░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░    ░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▒▒▒▒▒░ ░▒▒░    ",
            "    ░▒▒░ ░▒▒░ ░░░░                  ░▒▒░ ░░░░░░░░░░░░░░░░░░ ░▓▒░    ▒▒░ ░░░░░░░░░░░░░░░░░░░░░░░    ░░░░░░░░░░░░░░░░░░░░░░░ ░▒▒░    ",
            "    ░▒▒░ ░▒▒░ ░▒▒░                  ░▒▒▒                    ░▒▒░    ▒▒░                                                    ▒▒▒░    ",
            "    ▒▓▒░ ░▓▒░ ░▓▓░                  ░▒▒▒░░░░░░░░░░░░░░░░░░░░▒▒▒░    ▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░    ░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒░    ",
            "     ░░░  ░░   ░░                   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░    ░░░░░░░░░░░░░░░░░░░░░░░░░░░    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░    "
        ],
        // Head ASCII art
        asciiArt: [
            "                                                                                                    ",
            "                                                                                                    ",
            "                                          :=>(][[](>=:                                              ",
            "                                  =(@@@@@@@@@@@@@@@@@@@@@@@@]+                                      ",
            "                              <@@@@@@@@[*-.          .-+[@@@@@@@@>                                  ",
            "                          .]@@@@@#=                          ~#@@@@@)                               ",
            "                        *@@@@@=                                  +@@@@@:                            ",
            "                      >@@@@=                                        >@@@@.                          ",
            "                    -@@@@-                   .:---:.                  <@@@<                         ",
            "                   [@@@~              *@@@@@@@@@@@@@@@@@@@^             @@@}                        ",
            "                  @@@{           .)@@@@{>:             .>{@@@@)          {@@[                       ",
            "                 @@@)          )@@@(:                       .(@@@*        @@@=                      ",
            "                #@@]         [@@}                              .%@@+      ~@@@                      ",
            "               <@@%        *@@)                                  :@@#      [@@*                     ",
            "               @@@.       }@@              ->#@@@@@@#>:            [@@     +@@@                     ",
            "              >@@}       {@#          :]@@]+.        .*[@@(.        %@)    .@@@                     ",
            "              %@@+      (@%         (@].                  :%@+      :@@.   .@@@                     ",
            "              @@@.     :@@.       >@^                        #@.     {@<   :@@@                     ",
            "              @@@      }@]       #%                           <@     +@@   .@@@                     ",
            "              @@@      @@=      #{                             @[    +@%    (@@#                    ",
            "              @@@      @@.     ^@                              :@:   *@#     #@@]                   ",
            "              @@@=     @@.     %(                               @=   .@@+     @@@[                  ",
            "              <@@[     @@-     @^                               @-    .@@=     #@@%                 ",
            "               @@@.    {@(     @^                               {@     :@@>     (@@@:                ",
            "               ]@@}    =@@     %)                                }%      @@]     *@@@:               ",
            "                @@@~    #@(    ^@                                 )@.     #@}     +@@@              ",
            "                ~@@@    .@@:    %[                                 >@     .@@     -@@@              ",
            "                 ^@@@    *@@    .@<                               }@(   ~@@@<   %@@@@=              ",
            "                  <@@@    ^@@    -@>                             .@^    #@)    >@@}.                ",
            "                   <@@@    *@@:   :@)                            ~#@    .@@^   :@@@-                ",
            "                    >@@@.   =@@=    @{                           .@}   ~@@@*    ~@@@                ",
            "                     *@@@:   :@@^    @[                          ]@    .@@@   :@@@@%                ",
            "                      ~@@@+   .@@<    @+                         *@.    %@%   ^@@@@                 ",
            "                       .@@@*    @@>   @<                          @>   :@@      @@@<                ",
            "                        .@@@^   ~@@   @>                 ~>>>>><[@]     @@+    @@@]                 ",
            "                         .@@@~   @@~ .@-                [@.             >@#   .@@@                  ",
            "                          -@@@   }@< @[                 #]              #@}    @@@-                 ",
            "                           [@@+  {@^[@                  ]@  *{###{{{#%@@@[     *@@{                 ",
            "                           +@@[  @@#@                    @>@@@(][}}[]<+         @@@                 ",
            "                           -@@@ ^@@{                     +@@@                  <@@#                 ",
            "                           -@@@ @@^                       =@@   .=*++=~~~=+>[@@@@@.                 ",
            "                           =@@{[@#                         @@.~@@@@@@@@@@@@@@@@%:                    ",
            "                           <@@{@@.                         )@@@@@-     .                            ",
            "                           @@@@@-                          .@@@@)                                    ",
            "                          ~@@@@-                            ~@@@<                                    ",
            "                          {@@@.                              -@@#                                    ",
            "                         *@@@                                 @@@:                                   ",
            "                         @@@:                                 (@@)                                   ",
            "                        {@@(                                   @@@-                                  ",
            "                       [@@{                                    ~@@@                                 ",
            "                      (@@%                                      <@@@.                               ",
            "                     (@@@                                        *@@@~                              ",
            "                      (^                                           *+                               ",
            "                                                                                                    ",
            "                                                                                                    "
        ],
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
    
    // ASCII art buffer (between background and main text)
    let asciiArtBuffer;
    let asciiArtText = [];
    let asciiArtTypingIndex = 0;
    let asciiArtTypingSpeed = 50; // ms per character
    let asciiArtPhase = 0; // 0: not started, 1: typing on, 2: visible, 3: typing off, 4: hidden
    let asciiArtStartTime = 0;
    let asciiArtTypingOnDuration = 5000; // 5 seconds to type on
    let asciiArtVisibleDuration = 2000;  // 2 seconds visible
    let asciiArtTypingOffDuration = 4000; // 4 seconds to type off
    let asciiArtRandomOrder = []; // Random order for typing
    let asciiArtVisibleChars = []; // Track which chars are visible
    let asciiArtLastTypingTime = 0;
    
    // Title ASCII art animation (same system as head ASCII art)
    let titleAsciiText = [];
    let titleAsciiRandomOrder = [];
    let titleAsciiVisibleChars = [];
    let titleAsciiTypingIndex = 0;
    let titleAsciiLastTypingTime = 0;
    let titleAsciiPhase = 0; // 0: not started, 1: typing on, 2: visible, 3: typing off, 4: done
    
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
        
        // Initialize ASCII art buffer
        asciiArtBuffer = p.createGraphics(p.width, p.height);
        asciiArtBuffer.clear(); // Transparent background
        initializeAsciiArt();
        initializeTitleAsciiArt();
        
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
        
        // Draw ASCII art buffer on top of everything
        asciiArtBuffer.clear();
        drawAsciiArt();
        p.image(asciiArtBuffer, 0, 0);
        
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
    
    function initializeAsciiArt() {
        // Use CONFIG.asciiArt
        asciiArtText = [];
        for (let line of CONFIG.asciiArt) {
            asciiArtText.push(line.split(''));
        }
        setupAsciiArtRandomOrder();
    }
    
    function initializeTitleAsciiArt() {
        // Use CONFIG.titleAsciiArt
        titleAsciiText = [];
        for (let line of CONFIG.titleAsciiArt) {
            titleAsciiText.push(line.split(''));
        }
        setupTitleAsciiArtRandomOrder();
    }
    
    function setupAsciiArtRandomOrder() {
        asciiArtTypingIndex = 0;
        asciiArtPhase = 0;
        asciiArtStartTime = 0;
        
        // Create random order for typing
        asciiArtRandomOrder = [];
        let totalChars = 0;
        for (let y = 0; y < asciiArtText.length; y++) {
            for (let x = 0; x < asciiArtText[y].length; x++) {
                if (asciiArtText[y][x] !== ' ') {
                    asciiArtRandomOrder.push({x, y});
                    totalChars++;
                }
            }
        }
        
        // Shuffle the order
        for (let i = asciiArtRandomOrder.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [asciiArtRandomOrder[i], asciiArtRandomOrder[j]] = [asciiArtRandomOrder[j], asciiArtRandomOrder[i]];
        }
    }
    
    function setupTitleAsciiArtRandomOrder() {
        titleAsciiTypingIndex = 0;
        titleAsciiPhase = 0;
        
        // Create random order for typing
        titleAsciiRandomOrder = [];
        let totalChars = 0;
        for (let y = 0; y < titleAsciiText.length; y++) {
            for (let x = 0; x < titleAsciiText[y].length; x++) {
                if (titleAsciiText[y][x] !== ' ') {
                    titleAsciiRandomOrder.push({x, y});
                    totalChars++;
                }
            }
        }
        
        // Shuffle the order
        for (let i = titleAsciiRandomOrder.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [titleAsciiRandomOrder[i], titleAsciiRandomOrder[j]] = [titleAsciiRandomOrder[j], titleAsciiRandomOrder[i]];
        }
    }
    
    function drawAsciiArt() {
        if (!asciiArtText || asciiArtText.length === 0) return;
        
        asciiArtBuffer.push();
        asciiArtBuffer.fill(CONFIG.colors.grey);
        asciiArtBuffer.textFont('Courier New', fontSize * 0.6);
        
        // Center the ASCII art
        let startX = Math.floor((cols - asciiArtText[0].length) / 2);
        let startY = Math.floor((rows - asciiArtText.length) / 2);
        
        for (let i = 0; i < asciiArtText.length; i++) {
            let lineText = '';
            for (let j = 0; j < asciiArtText[i].length; j++) {
                if (asciiArtVisibleChars[i] && asciiArtVisibleChars[i][j]) {
                    lineText += asciiArtText[i][j];
                } else {
                    lineText += ' '; // Space for invisible characters
                }
            }
            
            if (lineText.trim().length > 0) {
                let x = startX * charWidth;
                let y = (startY + i) * charHeight;
                asciiArtBuffer.text(lineText, x, y);
            }
        }
        
        asciiArtBuffer.pop();
        updateAsciiArtAnimation();
    }
    
    function updateAsciiArtAnimation() {
        if (!asciiArtRandomOrder) return;
        
        let currentTime = millis();
        if (asciiArtPhase === 0) { // Typing on
            if (currentTime - asciiArtLastTypingTime > asciiArtTypingSpeed) {
                let charsToType = Math.floor(Math.random() * 21) + 10; // 10-30 chars per frame
                for (let i = 0; i < charsToType && asciiArtTypingIndex < asciiArtRandomOrder.length; i++) {
                    let pos = asciiArtRandomOrder[asciiArtTypingIndex];
                    if (!asciiArtVisibleChars[pos.y]) {
                        asciiArtVisibleChars[pos.y] = [];
                    }
                    asciiArtVisibleChars[pos.y][pos.x] = true;
                    asciiArtTypingIndex++;
                }
                asciiArtLastTypingTime = currentTime;
                
                if (asciiArtTypingIndex >= asciiArtRandomOrder.length) {
                    asciiArtPhase = 1; // Wait
                    asciiArtStartTime = currentTime;
                }
            }
        } else if (asciiArtPhase === 1) { // Wait
            if (currentTime - asciiArtStartTime > asciiArtVisibleDuration) {
                asciiArtPhase = 2; // Typing off
                asciiArtTypingIndex = asciiArtRandomOrder.length - 1;
            }
        } else if (asciiArtPhase === 2) { // Typing off
            if (currentTime - asciiArtLastTypingTime > asciiArtTypingSpeed) {
                let charsToType = Math.floor(Math.random() * 21) + 10; // 10-30 chars per frame
                for (let i = 0; i < charsToType && asciiArtTypingIndex >= 0; i++) {
                    let pos = asciiArtRandomOrder[asciiArtTypingIndex];
                    if (asciiArtVisibleChars[pos.y]) {
                        asciiArtVisibleChars[pos.y][pos.x] = false;
                    }
                    asciiArtTypingIndex--;
                }
                asciiArtLastTypingTime = currentTime;
                
                if (asciiArtTypingIndex < 0) {
                    asciiArtPhase = 0; // Reset to typing on
                    asciiArtTypingIndex = 0;
                    asciiArtVisibleChars = [];
                }
            }
        }
    }
    
    function drawAsciiTitle() {
        if (!titleAsciiText || titleAsciiText.length === 0) return;
        
        mainTextBuffer.push();
        mainTextBuffer.fill(CONFIG.colors.pureRed);
        mainTextBuffer.textFont('Courier New', fontSize * size);
        mainTextBuffer.textSize(fontSize * size);
        
        // Center the ASCII art
        let startX = Math.floor((cols - titleAsciiText[0].length) / 2);
        let startY = Math.floor((rows - titleAsciiText.length) / 2);
        
        for (let i = 0; i < titleAsciiText.length; i++) {
            let lineText = '';
            for (let j = 0; j < titleAsciiText[i].length; j++) {
                if (titleAsciiVisibleChars[i] && titleAsciiVisibleChars[i][j]) {
                    lineText += titleAsciiText[i][j];
                } else {
                    lineText += ' '; // Space for invisible characters
                }
            }
            
            if (lineText.trim().length > 0) {
                let x = startX * charWidth;
                let drawY = (startY + i) * charHeight;
                
                // Draw black background rectangle
                mainTextBuffer.fill(CONFIG.colors.background);
                mainTextBuffer.noStroke();
                mainTextBuffer.rect(x - 2, drawY - 2, lineText.length * charWidth + 4, charHeight + 4);
                
                // Draw ASCII art line
                mainTextBuffer.fill(CONFIG.colors.pureRed);
                mainTextBuffer.text(lineText, x, drawY);
            }
        }
        
        mainTextBuffer.pop();
        updateTitleAsciiAnimation();
    }
    
    function updateTitleAsciiAnimation() {
        if (!titleAsciiRandomOrder) return;
        
        let currentTime = millis();
        if (titleAsciiPhase === 0) { // Typing on
            if (currentTime - titleAsciiLastTypingTime > titleAsciiTypingSpeed) {
                let charsToType = Math.floor(Math.random() * 21) + 10; // 10-30 chars per frame
                for (let i = 0; i < charsToType && titleAsciiTypingIndex < titleAsciiRandomOrder.length; i++) {
                    let pos = titleAsciiRandomOrder[titleAsciiTypingIndex];
                    if (!titleAsciiVisibleChars[pos.y]) {
                        titleAsciiVisibleChars[pos.y] = [];
                    }
                    titleAsciiVisibleChars[pos.y][pos.x] = true;
                    titleAsciiTypingIndex++;
                }
                titleAsciiLastTypingTime = currentTime;
                
                if (titleAsciiTypingIndex >= titleAsciiRandomOrder.length) {
                    titleAsciiPhase = 1; // Complete
                }
            }
        }
    }
    
    function drawSettledContent() {
        // Draw main text to mainTextBuffer
        mainTextBuffer.push();
        mainTextBuffer.fill(CONFIG.colors.pureRed);
        mainTextBuffer.textFont('Courier New', fontSize);
        
        // Draw title as ASCII art
        drawAsciiTitle();
        
        // Draw event info
        let infoY = Math.floor(rows * 0.6);
        let infoText = [
            CONFIG.text.subtitle,
            CONFIG.text.date,
            CONFIG.text.address,
            CONFIG.text.description
        ];
        
        for (let i = 0; i < infoText.length; i++) {
            let y = infoY + i;
            if (y < rows) {
                let x = Math.floor(cols * 0.1);
                mainTextBuffer.text(infoText[i], x * charWidth, y * charHeight);
            }
        }
        
        mainTextBuffer.pop();
        
        // Show and position RSVP element
        if (rsvpElement) {
            rsvpElement.style.display = 'block';
            let rsvpX = Math.floor(cols * 0.1);
            let rsvpY = Math.floor(rows * 0.8);
            rsvpElement.style.left = (rsvpX * charWidth) + 'px';
            rsvpElement.style.top = (rsvpY * charHeight) + 'px';
        }
    }
    
    function drawCursor() {
        push();
        fill(CONFIG.colors.gold);
        noStroke();
        
        // Draw isosceles triangle pointing left
        let cursorX = mouseX;
        let cursorY = mouseY;
        let size = 8;
        
        triangle(
            cursorX - size, cursorY,           // Left point (sharpest)
            cursorX, cursorY - size/2,         // Top right
            cursorX, cursorY + size/2          // Bottom right
        );
        
        pop();
    }
    
    function drawFPSMonitor() {
        push();
        fill(CONFIG.colors.grey);
        textFont('Courier New', 12);
        textAlign(LEFT, TOP);
        text(`FPS: ${fps.toFixed(1)}`, 10, 10);
        pop();
    }
    
    function updateFPS() {
        frameCount++;
        let currentTime = millis();
        if (currentTime - lastFpsTime >= 1000) {
            fps = frameCount * 1000 / (currentTime - lastFpsTime);
            frameCount = 0;
            lastFpsTime = currentTime;
        }
    }
    
    p.windowResized = function() {
        p.resizeCanvas(p.windowWidth, p.windowHeight);
        initializeCharGrid();
        initializeTypingQueue();
        initializeSimpleTyping();
        initializeAsciiArt();
        initializeTitleAsciiArt();
        
        // Recreate buffers with new dimensions
        mainTextBuffer = p.createGraphics(p.width, p.height);
        asciiArtBuffer = p.createGraphics(p.width, p.height);
        
        // Update RSVP element position
        if (rsvpElement) {
            rsvpElement.style.fontSize = charHeight + 'px';
        }
    }
    
    p.mousePressed = function() {
        // Mouse press handling if needed
    };
    
    // Return the p5 instance
    return p;
}
