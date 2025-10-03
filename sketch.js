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
            lightGold: '#ff8800',
            white: '#ffffff'
        },
        timing: {
            introDuration: 12000, // 12 seconds total (extended for more content)
            scrollPhase: 5000,    // 0-5s: background typing
            typingPhase: 2000,    // 5-7s: title typing
            infoPhase: 4000,      // 7-11s: event info (extended for more text)
            rsvpPhase: 1000       // 11-12s: RSVP
        },
        text: {
            title: "THE FOLD",
            subtitle: ">>> YOU ARE INVITED <<<",
            date: "To a celebration",
            address: "8 years of The Fold",
            description: "at their new office",
            location: "LOCATION .....: 40w 100n Provo",
            time: "TIME .........: 6:30pm - 9:30pm on the 23rd of October",
            refreshments: "REFRESHMENTS .: drinks and refreshments provided",
            rsvpRequest: "RSVP .........: please rsvp",
            closing: ">>> we're looking forward to your initiation <<<",
            rsvpText: "→ RSVP"
        },
        // Title ASCII art
        titleAsciiArt: [
            "                                ▒▓▓▓▓▓▓▒   ▓▓   ▓█▓  ▓▓▓▓▓▓▓▒    ▓█▓  ▓█▓  ▓██                                               ",
            "                                   ██      ██   ██▓  ██          ███  ███  ███                                               ",
            "                                   █▓      ██▓▓▓██▓  ██▓▓▓▓▓▒    ███  ███  ███                                               ",
            "                                   █▓      ██   ██▓  ██          ██▓  ██▓  ███                                               ",
            "                                   ▓▓      ▓▓   ▓█▓  ▓▓▓▓▓▓▓▒    ███  ███  ███                                               ",
            "                                                                 ███  ███  ███                                               ",
            "▓██████████████████████████▓    ▓███████████████████████████▓    ██▓  ██▓  ███                   ▓██████████████████████████▓",
            "▓███▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█▓    ▓████▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█████    ███  ███  ███                   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█████",
            "███                             ████                     ████    ███  ███  ███                                            ███",
            "███  ██████████████████████▓    ███▓ ▓█████████████████▓ ████    ██▓  ██▓  ███                   ▓████████████████████▓▓  ███",
            "▓██ █████▓██▓██▓██▓██▓██▓██▓    ███▓ █████▓██▓██▓███████ ████    ███  ███  ███                   ▓██▓██▓██▓██▓██▓███████  ███",
            "▓██ ████                        ███▓ ████           ████ ████    ███  ███  ███                                      ████  ███",
            "███ ███████████████████████▓    ███▓ ███▓ ▓██████▓▓ ████ ████    ██▓  ██▓  ███                   ▓▓███████████████▓ ████  ███",
            "▓██ ▓▓██▓█▓▓█▓▓█▓▓█▓▓█▓▓█▓█▓    ███▓ ███▓ ▓████████ ████ ████    ███  ███  ███                   ▓▓██▓▓█▓▓█▓▓██████ ████  ███",
            "▓██                             ███▓ ███▓ ▓███ ████ ████ ████    ███  ███  ███                                 ████ ████  ███",
            "███ ▓▓█▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█▓    ███▓ ███▓ ▓████████ ████ ████    ██▓  ██▓  █████████████████▓    ▓█████████████████ ████  ███",
            "▓██ ███████████████████████▓    ███▓ ███▓ ▓███▓██▓▓ ████ ████    ███  ███  ▓█▓██▓▓█▓▓█▓▓█▓▓█▓    ▓██▓▓█▓▓█▓▓██▓███▓ ████  ███",
            "▓██ ████                        ███▓ ████           ████ ████    ███  ███                                           ████  ███",
            "███ ████  ██▓                   ███▓ ▓██▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ ████    ██▓  ▓██▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒    ▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█▓  ███",
            "▓██ ████  ███                   ████                     ████    ███                                                      ███",
            "▓██ ████  ███                   ███████▓█████████████████████    ███████████████████████████▓    ▓███████████████████████████",
            "▓█▓ ▓▓█▓  ██▓                   ▓██▓██▓██▓▓█▓▓█▓▓█▓▓█▓▓████▓▓    ▓██▓█▓▓█▓▓█▓▓█▓▓█▓▓█▓▓█▓▓██▓    ▓██▓▓█▓▓█▓▓█▓▓█▓▓█▓▓█▓█████▓"
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
    let mainTextBufferNeedsUpdate = true;
    
    // Grid system
    let charWidth, charHeight;
    let cols, rows;
    let fontSize;
    
    // Character grid system
    let gridCols, gridRows;
    let backgroundLastTypingTime = 0;
    let backgroundTypingSpeed = 50; // ms per character (same as ASCII art)
    let backgroundRandomOrder = []; // Random order for typing
    let backgroundVisibleChars = []; // Track which chars are visible
    let backgroundTypingIndex = 0;
    let backgroundPhase = 0; // 0: typing on, 1: visible, 2: typing off
    let backgroundStartTime = 0;
    let backgroundTypingOnDuration = 5000; // 5 seconds to type on
    let backgroundVisibleDuration = 3000;  // 3 seconds visible
    let backgroundTypingOffDuration = 2000; // 2 seconds to type off
    let backgroundText = []; // Store the background text as 2D array
    
    // Individual character lifecycle system
    let characterLifecycles = []; // Store individual character timing
    let characterCycleDuration = 8000; // Total cycle time (on + visible + off)
    let characterTypingOnDuration = 1500; // Individual character typing on time
    let characterVisibleDuration = 4000; // Individual character visible time
    let characterTypingOffDuration = 1500; // Individual character typing off time
    let characterFadeDuration = 500; // Fade in/out time
    
    // Main text buffer (separate layer)
    let mainTextBuffer;
    
    // ASCII art buffer (between background and main text)
    let asciiArtBuffer;
    
    // Title ASCII art buffer (above event info)
    let titleAsciiBuffer;
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
    let titleAsciiTypingSpeed = 50; // ms per character
    
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
    let titleX, titleY, infoX;
    let infoLines = [];
    let infoStartY;
    
    // Color transitions (removed - text types directly as red/gold)
    
    // Mouse interaction
    let mouseProximity = 100; // pixels
    let scrambledChars = {};
    let scrambleTimeouts = {};
    
    // Background noise movement
    let noiseScale = 0.09; // Controls noise frequency
    let noiseStrength = 5; // Multiplier for character dimensions (5x = ~30-40px movement)
    let noiseSpeed = 0.01; // Speed of time-based animation
    
    // Particle weights and physics
    let particleWeights = []; // Store weights for each background character
    let baseGrey = '#4a4a4a'; // Base grey color
    let brightGrey = '#8a8a8a'; // Brighter grey for weighted particles
    let mouseRepelStrength = 0.9; // How much particles are repelled by mouse
    
    // Retro ASCII ornamentation
    let ornamentChars = ['◊', '♦', '▲', '▼', '◄', '►', '◈', '◉', '◯', '◌', '◍', '◎', '◐', '◑', '◒', '◓', '◔', '◕', '◖', '◗', '◘', '◙', '◚', '◛', '◜', '◝', '◞', '◟', '◠', '◡', '◢', '◣', '◤', '◥', '◦', '◧', '◨', '◩', '◪', '◫', '◬', '◭', '◮', '◯', '◰', '◱', '◲', '◳', '◴', '◵', '◶', '◷', '◸', '◹', '◺', '◻', '◼', '◽', '◾', '◿', '☀', '☁', '☂', '☃', '☄', '★', '☆', '☇', '☈', '☉', '☊', '☋', '☌', '☍', '☎', '☏', '☐', '☑', '☒', '☓', '☔', '☕', '☖', '☗', '☘', '☙', '☚', '☛', '☜', '☝', '☞', '☟', '☠', '☡', '☢', '☣', '☤', '☥', '☦', '☧', '☨', '☩', '☪', '☫', '☬', '☭', '☮', '☯', '☰', '☱', '☲', '☳', '☴', '☵', '☶', '☷', '☸', '☹', '☺', '☻', '☼', '☽', '☾', '☿', '♀', '♂', '♁', '♃', '♄', '♅', '♆', '♇', '♈', '♉', '♊', '♋', '♌', '♍', '♎', '♏', '♐', '♑', '♒', '♓', '♔', '♕', '♖', '♗', '♘', '♙', '♚', '♛', '♜', '♝', '♞', '♟', '♠', '♡', '♢', '♣', '♤', '♥', '♦', '♧', '♨', '♩', '♪', '♫', '♬', '♭', '♮', '♯', '♰', '♱', '♲', '♳', '♴', '♵', '♶', '♷', '♸', '♹', '♺', '♻', '♼', '♽', '♾', '♿', '⚀', '⚁', '⚂', '⚃', '⚄', '⚅', '⚆', '⚇', '⚈', '⚉', '⚊', '⚋', '⚌', '⚍', '⚎', '⚏', '⚐', '⚑', '⚒', '⚓', '⚔', '⚕', '⚖', '⚗', '⚘', '⚙', '⚚', '⚛', '⚜', '⚝', '⚞', '⚟', '⚠', '⚡', '⚢', '⚣', '⚤', '⚥', '⚦', '⚧', '⚨', '⚩', '⚪', '⚫', '⚬', '⚭', '⚮', '⚯', '⚰', '⚱', '⚲', '⚳', '⚴', '⚵', '⚶', '⚷', '⚸', '⚹', '⚺', '⚻', '⚼', '⚽', '⚾', '⚿', '⛀', '⛁', '⛂', '⛃', '⛄', '⛅', '⛆', '⛇', '⛈', '⛉', '⛊', '⛋', '⛌', '⛍', '⛎', '⛏', '⛐', '⛑', '⛒', '⛓', '⛔', '⛕', '⛖', '⛗', '⛘', '⛙', '⛚', '⛛', '⛜', '⛝', '⛞', '⛟', '⛠', '⛡', '⛢', '⛣', '⛤', '⛥', '⛦', '⛧', '⛨', '⛩', '⛪', '⛫', '⛬', '⛭', '⛮', '⛯', '⛰', '⛱', '⛲', '⛳', '⛴', '⛵', '⛶', '⛷', '⛸', '⛹', '⛺', '⛻', '⛼', '⛽', '⛾', '⛿', '✀', '✁', '✂', '✃', '✄', '✅', '✆', '✇', '✈', '✉', '✊', '✋', '✌', '✍', '✎', '✏', '✐', '✑', '✒', '✓', '✔', '✕', '✖', '✗', '✘', '✙', '✚', '✛', '✜', '✝', '✞', '✟', '✠', '✡', '✢', '✣', '✤', '✥', '✦', '✧', '✨', '✩', '✪', '✫', '✬', '✭', '✮', '✯', '✰', '✱', '✲', '✳', '✴', '✵', '✶', '✷', '✸', '✹', '✺', '✻', '✼', '✽', '✾', '✿', '❀', '❁', '❂', '❃', '❄', '❅', '❆', '❇', '❈', '❉', '❊', '❋', '❌', '❍', '❎', '❏', '❐', '❑', '❒', '❓', '❔', '❕', '❖', '❗', '❘', '❙', '❚', '❛', '❜', '❝', '❞', '❟', '❠', '❡', '❢', '❣', '❤', '❥', '❦', '❧', '❨', '❩', '❪', '❫', '❬', '❭', '❮', '❯', '❰', '❱', '❲', '❳', '❴', '❵', '❶', '❷', '❸', '❹', '❺', '❻', '❼', '❽', '❾', '❿', '➀', '➁', '➂', '➃', '➄', '➅', '➆', '➇', '➈', '➉', '➊', '➋', '➌', '➍', '➎', '➏', '➐', '➑', '➒', '➓', '➔', '➕', '➖', '➗', '➘', '➙', '➚', '➛', '➜', '➝', '➞', '➟', '➠', '➡', '➢', '➣', '➤', '➥', '➦', '➧', '➨', '➩', '➪', '➫', '➬', '➭', '➮', '➯', '➰', '➱', '➲', '➳', '➴', '➵', '➶', '➷', '➸', '➹', '➺', '➻', '➼', '➽', '➾', '➿'];
    let ornamentPositions = []; // Store ornament positions
    let ornamentDensity = 0.02; // How many ornaments per grid cell
    
    // Post-processing (disabled for performance)
    // let scanlineOffset = 0;
    // let enableBloom = false;
    // let enableScanlines = false;
    // let enableBlur = false;
    
    // Cursor
    let cursorSize = 18;
    
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
        charHeight = fontSize; // Line height
        cols = p.floor(p.width / charWidth);
        rows = p.floor(p.height / charHeight);
        
        // Initialize character grid
        initializeCharGrid();
        
        // Initialize main text buffer
        mainTextBuffer = p.createGraphics(p.width, p.height);
        mainTextBuffer.clear(); // Transparent background
        
        // Initialize ASCII art buffer
        asciiArtBuffer = p.createGraphics(p.width, p.height);
        
        // Initialize title ASCII art buffer
        titleAsciiBuffer = p.createGraphics(p.width, p.height);
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
        
        // Handle intro sequence phases and main text (on separate buffer)
        if (!isIntroComplete) {
            // Clear main text buffer before drawing new content
            mainTextBuffer.clear();
            
            // Hide RSVP element during intro phases
            if (rsvpElement && introPhase < 3) {
                rsvpElement.style.display = 'none';
            }
            handleIntroSequence(elapsed);
        } else {
            // In settled state, clear and redraw main text buffer
            mainTextBuffer.clear();
            drawIntroContent();
        }
        
        // Draw layers in correct order (bottom to top):
        // 1. Background code (already drawn)
        // 2. ASCII art head buffer (always visible, cycling animation)
        asciiArtBuffer.clear();
        drawAsciiArt();
        p.image(asciiArtBuffer, 0, 0);
        
        // 3. Event info (main text buffer)
        p.image(mainTextBuffer, 0, 0);
        
        // 4. Title ASCII art buffer (above event info)
        if (introPhase >= 1) {
            titleAsciiBuffer.clear();
            drawAsciiTitle();
            p.image(titleAsciiBuffer, 0, 0);
        }
        
        // Draw cursor
        drawCursor();
        
        // Draw FPS monitor
        drawFPSMonitor();
        
        // Update FPS
        updateFPS();
        
        // Post-processing disabled for performance
    };

    function initializeCharGrid() {
        // Initialize character grid
        gridCols = cols;
        gridRows = rows;
        
        // Initialize typing queue with new random system
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
    
    function initializeTypingQueue() {
        // Initialize background text as 2D array
        backgroundText = [];
        particleWeights = []; // Initialize particle weights
        characterLifecycles = []; // Initialize character lifecycles
        for (let y = 0; y < gridRows; y++) {
            backgroundText[y] = [];
            particleWeights[y] = [];
            characterLifecycles[y] = [];
            for (let x = 0; x < gridCols; x++) {
                backgroundText[y][x] = ' ';
                particleWeights[y][x] = p.random(0.1, 1.0); // Random weight between 0.1 and 1.0
                
                // Initialize individual character lifecycle
                characterLifecycles[y][x] = {
                    phase: 0, // 0: typing on, 1: visible, 2: typing off, 3: hidden
                    startTime: p.random(0, characterCycleDuration), // Random start time
                    opacity: 0
                };
            }
        }
        
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
        
        // Fill the background text array
        for (let pos of positions) {
            for (let i = 0; i < pos.text.length; i++) {
                if (pos.x + i < gridCols) {
                    backgroundText[pos.y][pos.x + i] = pos.text[i];
                }
            }
        }
        
        // Create random order for typing
        backgroundRandomOrder = [];
        for (let y = 0; y < gridRows; y++) {
            for (let x = 0; x < gridCols; x++) {
                if (backgroundText[y][x] !== ' ') {
                    backgroundRandomOrder.push({x, y});
                }
            }
        }
        
        // Shuffle the order
        for (let i = backgroundRandomOrder.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [backgroundRandomOrder[i], backgroundRandomOrder[j]] = [backgroundRandomOrder[j], backgroundRandomOrder[i]];
        }
        
        // Initialize visible chars array
        backgroundVisibleChars = [];
        for (let y = 0; y < gridRows; y++) {
            backgroundVisibleChars[y] = [];
            for (let x = 0; x < gridCols; x++) {
                backgroundVisibleChars[y][x] = false;
            }
        }
        
        // Reset animation state
        backgroundTypingIndex = 0;
        backgroundPhase = 0;
        backgroundStartTime = 0;
    }
    
    function initializeSimpleTyping() {
        // Set up simple text positions
        titleX = 2; // Start at column 2 (very close to left edge)
        titleY = p.floor(p.height * 0.07) / charHeight; // Higher up in top quarter
        
        infoLines = [
            CONFIG.text.subtitle,
            CONFIG.text.date,
            CONFIG.text.address,
            CONFIG.text.description,
            "", // Empty line before location
            CONFIG.text.location,
            CONFIG.text.time,
            CONFIG.text.refreshments,
            CONFIG.text.rsvpRequest,
            "", // Empty line before closing
            CONFIG.text.closing
        ];
        
        // Calculate center position for info text
        let maxLineLength = 0;
        for (let line of infoLines) {
            maxLineLength = Math.max(maxLineLength, line.length);
        }
        let infoTextWidth = maxLineLength * charWidth;
        infoX = p.floor((p.width - infoTextWidth) / 2 / charWidth); // Center horizontally
        infoStartY = p.floor(p.height * 0.55) / charHeight; // Move up more to prevent RSVP overlap
        
        // Reset typing indices
        titleTypingIndex = 0;
        infoTypingIndex = 0;
        titleText = "";
        infoText = "";
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
    
    function drawCharGrid() {
        p.push();
        p.textAlign(p.LEFT, p.TOP);
        p.fill(CONFIG.colors.grey);
        p.textFont('Courier New', fontSize);
        
        // Draw characters that are visible (using individual lifecycles)
        for (let y = 0; y < gridRows; y++) {
            for (let x = 0; x < gridCols; x++) {
                // Check both old system and new individual lifecycle system
                let isVisible = (backgroundVisibleChars[y] && backgroundVisibleChars[y][x]) || 
                               (characterLifecycles[y] && characterLifecycles[y][x] && characterLifecycles[y][x].opacity > 0);
                
                if (isVisible) {
                    let screenX = x * charWidth;
                    let screenY = y * charHeight;
                    let displayChar = backgroundText[y][x];
                    let weight = particleWeights[y] && particleWeights[y][x] ? particleWeights[y][x] : 0.5;
                    
                    // Apply mouse scramble effect to background code (works from phase 0)
                    let scrambled = applyMouseScrambleToCell(x, y, screenX, screenY);
                    if (scrambled) {
                        displayChar = scrambled;
                    }
                    
                    // Add organic noise movement to background characters
                    // Use unique time offsets for each character to break synchronization
                    let timeOffsetX = x * 0.1 + y * 0.07; // Unique offset based on position
                    let timeOffsetY = x * 0.13 + y * 0.11; // Different offset for Y to create independent movement
                    let noiseX = p.noise(x * noiseScale, y * noiseScale, p.frameCount * noiseSpeed + timeOffsetX) - 0.5;
                    let noiseY = p.noise(x * noiseScale + 100, y * noiseScale + 100, p.frameCount * noiseSpeed + timeOffsetY) - 0.5;
                    let offsetX = noiseX * noiseStrength * charWidth;  // Scale by character width
                    let offsetY = noiseY * noiseStrength * charHeight; // Scale by character height
                    
                    // Add mouse repulsion effect and size scaling (works from phase 0)
                    let mouseX = p.mouseX;
                    let mouseY = p.mouseY;
                    let distance = p.dist(screenX + offsetX, screenY + offsetY, mouseX, mouseY);
                    let repelRadius = mouseProximity * 2; // Larger repulsion radius
                    let sizeScale = 1.0; // Default size
                    
                    if (distance < repelRadius && distance > 0) {
                        let repelForce = (repelRadius - distance) / repelRadius;
                        let repelX = (screenX + offsetX - mouseX) / distance;
                        let repelY = (screenY + offsetY - mouseY) / distance;
                        
                        // Apply repulsion
                        offsetX += repelX * repelForce * mouseRepelStrength * weight * charWidth;
                        offsetY += repelY * repelForce * mouseRepelStrength * weight * charHeight;
                        
                        // Calculate size scaling based on distance and weight
                        let maxScale = 1.4; // Maximum 2x size
                        let sizeForce = (repelRadius - distance) / repelRadius;
                        sizeScale = 1.0 + (sizeForce * (maxScale - 1.0) * weight);
                    }
                    
                    // Set color based on weight (heavier = brighter) and opacity
                    let colorIntensity = p.lerp(0, 1, weight);
                    let currentColor = p.lerpColor(p.color(baseGrey), p.color(brightGrey), colorIntensity);
                    
                    // Apply opacity from individual lifecycle
                    let opacity = 1;
                    if (characterLifecycles[y] && characterLifecycles[y][x]) {
                        opacity = characterLifecycles[y][x].opacity;
                    }
                    
                    p.fill(p.red(currentColor), p.green(currentColor), p.blue(currentColor), opacity * 255);
                    
                    // Apply size scaling
                    p.push();
                    p.textSize(fontSize * sizeScale);
                    
                    // Draw character with noise offset, mouse repulsion, and size scaling
                    p.text(displayChar, screenX + offsetX, screenY + offsetY);
                    p.pop();
                }
            }
        }
        
        // Update typing animation
        updateCharGridTyping();
        updateIndividualCharacterLifecycles();
        p.pop();
    }
    
    function updateCharGridTyping() {
        if (!backgroundRandomOrder) return;
        
        let currentTime = p.millis();
        if (backgroundPhase === 0) { // Typing on
            if (currentTime - backgroundLastTypingTime > backgroundTypingSpeed) {
                let charsToType = Math.floor(Math.random() * 21) + 10; // 10-30 chars per frame
                for (let i = 0; i < charsToType && backgroundTypingIndex < backgroundRandomOrder.length; i++) {
                    let pos = backgroundRandomOrder[backgroundTypingIndex];
                    if (!backgroundVisibleChars[pos.y]) {
                        backgroundVisibleChars[pos.y] = [];
                    }
                    backgroundVisibleChars[pos.y][pos.x] = true;
                    backgroundTypingIndex++;
                }
                backgroundLastTypingTime = currentTime;
                
                if (backgroundTypingIndex >= backgroundRandomOrder.length) {
                    backgroundPhase = 1; // Wait
                    backgroundStartTime = currentTime;
                }
            }
        } else if (backgroundPhase === 1) { // Wait
            if (currentTime - backgroundStartTime > backgroundVisibleDuration) {
                backgroundPhase = 2; // Typing off
                backgroundTypingIndex = backgroundRandomOrder.length - 1;
            }
        } else if (backgroundPhase === 2) { // Typing off
            if (currentTime - backgroundLastTypingTime > backgroundTypingSpeed) {
                let charsToType = Math.floor(Math.random() * 21) + 10; // 10-30 chars per frame
                for (let i = 0; i < charsToType && backgroundTypingIndex >= 0; i++) {
                    let pos = backgroundRandomOrder[backgroundTypingIndex];
                    if (backgroundVisibleChars[pos.y]) {
                        backgroundVisibleChars[pos.y][pos.x] = false;
                    }
                    backgroundTypingIndex--;
                }
                backgroundLastTypingTime = currentTime;
                
                if (backgroundTypingIndex < 0) {
                    backgroundPhase = 0; // Restart
                    backgroundTypingIndex = 0;
                    // Reshuffle for next cycle
                    for (let i = backgroundRandomOrder.length - 1; i > 0; i--) {
                        const j = Math.floor(Math.random() * (i + 1));
                        [backgroundRandomOrder[i], backgroundRandomOrder[j]] = [backgroundRandomOrder[j], backgroundRandomOrder[i]];
                    }
                }
            }
        }
    }
    
    function updateIndividualCharacterLifecycles() {
        let currentTime = p.millis();
        
        for (let y = 0; y < gridRows; y++) {
            for (let x = 0; x < gridCols; x++) {
                if (!characterLifecycles[y] || !characterLifecycles[y][x]) continue;
                
                let char = characterLifecycles[y][x];
                let cycleTime = (currentTime - char.startTime) % characterCycleDuration;
                
                if (cycleTime < characterTypingOnDuration) {
                    // Typing on phase
                    char.phase = 0;
                    char.opacity = p.map(cycleTime, 0, characterTypingOnDuration, 0, 1);
                } else if (cycleTime < characterTypingOnDuration + characterVisibleDuration) {
                    // Visible phase
                    char.phase = 1;
                    char.opacity = 1;
                } else if (cycleTime < characterTypingOnDuration + characterVisibleDuration + characterTypingOffDuration) {
                    // Typing off phase
                    char.phase = 2;
                    let offTime = cycleTime - characterTypingOnDuration - characterVisibleDuration;
                    char.opacity = p.map(offTime, 0, characterTypingOffDuration, 1, 0);
                } else {
                    // Hidden phase
                    char.phase = 3;
                    char.opacity = 0;
                }
            }
        }
    }
    
    function applyMouseScrambleToCell(x, y, screenX, screenY) {
        let mouseX = p.mouseX;
        let mouseY = p.mouseY;
        let distance = p.dist(screenX, screenY, mouseX, mouseY);
        
        // Get particle weight and calculate threshold
        let weight = particleWeights[y] && particleWeights[y][x] ? particleWeights[y][x] : 0.5;
        let threshold = mouseProximity * weight; // Heavier particles have larger scramble radius
        
        if (distance < threshold) {
            let cellKey = `${x}-${y}`;
            
            // Initialize scramble tracking if needed
            if (!window.scrambleCells) {
                window.scrambleCells = {};
            }
            
            if (!window.scrambleCells[cellKey]) {
                window.scrambleCells[cellKey] = {
                    scrambleChar: randomScrambleChar(),
                    scrambleTimeout: null
                };
                
                // Set timeout to revert character
                if (window.scrambleCells[cellKey].scrambleTimeout) {
                    clearTimeout(window.scrambleCells[cellKey].scrambleTimeout);
                }
                window.scrambleCells[cellKey].scrambleTimeout = setTimeout(() => {
                    delete window.scrambleCells[cellKey];
                }, 300);
            }
            return window.scrambleCells[cellKey].scrambleChar;
        }
        return null;
    }
    
    function randomScrambleChar() {
        let scrambleChars = ['#', '@', '$', '%', '&', '*', '+', '=', '~', '^'];
        return scrambleChars[p.floor(p.random(scrambleChars.length))];
    }
    
    function handleIntroSequence(elapsed) {
        if (introPhase === 0 && elapsed > 5000) { // Wait 5 seconds before showing title
            // Start typing phase
            console.log("Transitioning to Phase 1 - Title typing");
            introPhase = 1;
            phaseStartTime = currentTime;
            initializeSimpleTyping();
        } else if (introPhase === 1) {
            // Wait for title typing to complete
            if (titleTypingIndex >= CONFIG.text.title.length) {
                console.log("Transitioning to Phase 2 - Info typing");
                introPhase = 2;
                phaseStartTime = currentTime;
            }
        } else if (introPhase === 2) {
            // Wait for info typing to complete
            let totalInfoChars = infoLines.reduce((sum, line) => sum + line.length + 1, 0); // +1 for newline
            if (infoTypingIndex >= totalInfoChars) {
                // Intro complete - stay in this phase
                console.log("Transitioning to Phase 3 - Complete");
                introPhase = 3;
                isIntroComplete = true;
                phaseStartTime = currentTime;
            }
        }
        
        // Handle typing animation
        if (introPhase === 1) {
            handleTitleTyping();
        } else if (introPhase === 2) {
            handleTitleTyping();
            handleInfoTyping();
        } else if (introPhase >= 3) {
            // Keep everything visible in final phase
            handleTitleTyping();
            handleInfoTyping();
        }
        
        // Draw content based on phase
        drawIntroContent();
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
            let currentLine = 0;
            let charCount = 0;
            
            // Find which line we're currently typing
            for (let i = 0; i < infoLines.length; i++) {
                if (infoTypingIndex < charCount + infoLines[i].length + 1) { // +1 for newline
                    currentLine = i;
                    break;
                }
                charCount += infoLines[i].length + 1; // +1 for newline
            }
            
            if (currentLine < infoLines.length) {
                let lineText = infoLines[currentLine];
                let lineIndex = infoTypingIndex - charCount;
                
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
    
    function drawIntroContent() {
        // Draw info text
        if (introPhase >= 2) {
            // In phase 3, use the full info text from CONFIG
            if (introPhase >= 3) {
                let fullInfoText = infoLines.join('\n');
                drawSimpleText(fullInfoText, infoX, infoStartY, CONFIG.colors.background, 1, CONFIG.colors.pureRed);
            } else if (infoText.length > 0) {
                drawSimpleText(infoText, infoX, infoStartY, CONFIG.colors.background, 1, CONFIG.colors.pureRed);
            }
        }
        
        // Draw RSVP text
        if (introPhase >= 3) {
            // Show and position HTML RSVP element instead of drawing to buffer
            if (rsvpElement) {
                let rsvpX = infoX * charWidth; // Use same X as info text (centered)
                let rsvpY = (infoStartY + infoLines.length + 1) * charHeight; // Position below all info text
                
                rsvpElement.style.left = rsvpX + 'px';
                rsvpElement.style.top = rsvpY + 'px';
                rsvpElement.style.fontSize = fontSize + 'px';
                rsvpElement.style.display = 'block';
                rsvpElement.style.zIndex = '1000'; // Ensure it's on top
            }
        } else {
            // Hide RSVP during intro phases
            if (rsvpElement) {
                rsvpElement.style.display = 'none';
            }
        }
    }
    
    function drawSimpleText(text, startX, startY, color, size, backgroundColor = null) {
        // Draw text with background
        let lines = text.split('\n');
        
        for (let i = 0; i < lines.length; i++) {
            let x = startX * charWidth;
            let y = (startY + i) * charHeight;
            
            mainTextBuffer.push();
            mainTextBuffer.textAlign(mainTextBuffer.LEFT, mainTextBuffer.CENTER);
            mainTextBuffer.textSize(fontSize);
            mainTextBuffer.textFont('Courier New', fontSize);
            
            // Draw background rectangle
            if (backgroundColor) {
                mainTextBuffer.fill(p.color(backgroundColor));
            } else {
                mainTextBuffer.fill(p.color(CONFIG.colors.background));
            }
            mainTextBuffer.noStroke();
            mainTextBuffer.rect(x - 2, y - 2, lines[i].length * charWidth + 4, charHeight + 4);
            
            // Draw text (centered vertically in the rectangle)
            mainTextBuffer.fill(p.color(color));
            mainTextBuffer.noStroke();
            mainTextBuffer.text(lines[i], x, y + charHeight / 2);
            
            mainTextBuffer.pop();
        }
    }
    
    function drawAsciiArt() {
        if (!asciiArtText || asciiArtText.length === 0) return;
        
        asciiArtBuffer.push();
        asciiArtBuffer.fill(CONFIG.colors.gold);
        asciiArtBuffer.textFont('Courier New', fontSize);
        
        // Center the ASCII art horizontally (calculate based on actual text width)
        let maxLineLength = 0;
        for (let line of asciiArtText) {
            maxLineLength = Math.max(maxLineLength, line.length);
        }
        let textWidthPixels = maxLineLength * charWidth;
        let startX = Math.floor((p.width - textWidthPixels) / 2) / charWidth;
        let startY = Math.floor((p.height - asciiArtText.length * charHeight) / 2) / charHeight;
        
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
        
        let currentTime = p.millis();
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
        
        titleAsciiBuffer.push();
        titleAsciiBuffer.fill(CONFIG.colors.pureRed);
        titleAsciiBuffer.textFont('Courier New', fontSize);
        titleAsciiBuffer.textSize(fontSize);
        
        // Calculate scaling to fit within page width while maintaining aspect ratio
        let maxLineLength = 0;
        for (let line of titleAsciiText) {
            maxLineLength = Math.max(maxLineLength, line.length);
        }
        
        // Calculate scale factor to fit within 90% of page width
        let maxWidthPixels = p.width * 0.9;
        let originalWidthPixels = maxLineLength * charWidth;
        let scaleFactor = Math.min(1, maxWidthPixels / originalWidthPixels);
        
        // Calculate scaled dimensions
        let scaledCharWidth = charWidth * scaleFactor;
        let scaledCharHeight = charHeight * scaleFactor;
        let scaledTextWidth = maxLineLength * scaledCharWidth;
        
        // Center the scaled ASCII art
        let startX = (p.width - scaledTextWidth) / 2;
        let startY = p.height * 0.15; // Higher up in top quarter
        
        // Set scaled font size
        titleAsciiBuffer.textSize(fontSize * scaleFactor);
        
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
                let x = startX;
                let drawY = startY + (i * scaledCharHeight);
                
                // Draw black background rectangle
                // mainTextBuffer.fill(CONFIG.colors.background);
                // mainTextBuffer.noStroke();
                // mainTextBuffer.rect(x - 2, drawY - 2, lineText.length * scaledCharWidth + 4, scaledCharHeight + 4);
                
                // Draw ASCII art line
                titleAsciiBuffer.fill(CONFIG.colors.pureRed);
                titleAsciiBuffer.text(lineText, x, drawY);
            }
        }
        
        titleAsciiBuffer.pop();
        updateTitleAsciiAnimation();
    }
    
    function updateTitleAsciiAnimation() {
        if (!titleAsciiRandomOrder) return;
        
        let currentTime = p.millis();
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
        
        // Draw event info with black text on red background
        let infoY = Math.floor(p.height * 0.65) / charHeight; // Bottom third of page
        let infoText = [
            CONFIG.text.subtitle,
            CONFIG.text.date,
            CONFIG.text.address,
            CONFIG.text.description
        ];
        
        // Draw each line with black text on red background
        for (let i = 0; i < infoText.length; i++) {
            let x = titleX * charWidth;
            let y = (infoY + i * 2) * charHeight;
            
            // Draw red background rectangle
            mainTextBuffer.fill(CONFIG.colors.pureRed);
            mainTextBuffer.noStroke();
            mainTextBuffer.rect(x - 2, y - 2, infoText[i].length * charWidth + 4, charHeight + 4);
            
            // Draw black text (align with background rectangle)
            mainTextBuffer.fill(CONFIG.colors.background);
            mainTextBuffer.text(infoText[i], x, y + charHeight - 2); // Align with background
        }
        
        mainTextBuffer.pop();
        
        // Show and position RSVP element
        if (rsvpElement) {
            rsvpElement.style.display = 'block';
            // Position RSVP below info text (centered)
            let rsvpX = infoX * charWidth;
            let rsvpY = (infoStartY + infoLines.length + 1) * charHeight;
            rsvpElement.style.left = rsvpX + 'px';
            rsvpElement.style.top = rsvpY + 'px';
        }
    }
    
    function drawCursor() {
        p.push();
        p.fill(CONFIG.colors.white);
        p.noStroke();
        
        // Draw isosceles triangle pointing left
        let cursorX = p.mouseX;
        let cursorY = p.mouseY;
        let size = 8;
        
        p.triangle(
            cursorX - size, cursorY,           // Left point (sharpest)
            cursorX, cursorY - size/2,         // Top right
            cursorX, cursorY + size/2          // Bottom right
        );
        
        p.pop();
    }
    
    function drawFPSMonitor() {
        p.push();
        p.fill(CONFIG.colors.grey);
        p.textFont('Courier New', 12);
        p.textAlign(p.LEFT, p.TOP);
        p.text(`FPS: ${fps.toFixed(1)}`, 10, 10);
        p.pop();
    }
    
    function updateFPS() {
        frameCount++;
        let currentTime = p.millis();
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
        titleAsciiBuffer = p.createGraphics(p.width, p.height);
        
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
