
const btn = document.getElementById('translateButton');
const frenchcase = document.getElementById('frenchOutput')
const jsonFileUrl1 = 'en-fr.json';

let tokenizer = null;
let pad_token = null;
let sos_token = null;
let eos_token = null;
let max_gen = 10;
let running = false


// Async function to fetch and read a JSON file
async function loadJsonFile(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error('Error fetching the JSON file:', error);
        return null; // Return null or appropriate default value in case of an error
    }
}

window.onload = async function() {
    tokenToIndexMapping = await loadJsonFile(jsonFileUrl1);
    console.log("JSON Data Loaded:", tokenToIndexMapping);
    session = await ort.InferenceSession.create('model_1_quantized.onnx');
    console.log("Session created:", session);
        // Example Usage
    tokenizer = new BPE_Tokenizer();
    tokenizer.loadVocab(tokenToIndexMapping); // Replace with your mapping
    
    let tokenIds = tokenizer.tokenize("Example text to tokenize");
    console.log("Token IDs:", tokenIds);
    
    let detokenizedText = tokenizer.detokenize(tokenIds);
    console.log("Detokenized Text:", detokenizedText);

    pad_token = tokenizer.tokenize("<PAS>");
    sos_token = tokenizer.tokenize("<EOS>");
    eos_token = tokenizer.tokenize("<EOS>");

};
// Async function to fetch and read a JSON file
class BPE_Tokenizer {
    constructor() {
        this.vocab = new Set();
        this.tokenToIndex = {};
        this.indexToToken = {};
    }

    tokenize(text) {
        let tokens = [];
        let words = text.split(/\s+/);
        for (let word of words) {
            let subwords = this.getSubwords(word + '</w>');
            for (let sw of subwords) {
                tokens.push(this.tokenToIndex[sw] ?? this.tokenToIndex['</u>']);
            }
        }
        tokens.map(item => BigInt(item));
        return tokens;
    }

    getSubwords(word) {
        let subwords = [];
        while (word) {
            let subword = this.findLongestSubword(word);
            if (!subword) {
                subwords.push('</u>');
                break;
            }
            subwords.push(subword);
            word = word.slice(subword.length);
        }
        return subwords;
    }

    findLongestSubword(word) {
        for (let i = word.length; i > 0; i--) {
            let possibleSubword = word.slice(0, i);
            if (this.vocab.has(possibleSubword)) {
                return possibleSubword;
            }
        }
        return null;
    }

    detokenize(tokenIds) {
        let words = [];
        let currentWord = '';
        for (let token_id of tokenIds) {
            let token = this.indexToToken[token_id] ?? '</u>';
            if (token === '</w>') {
                words.push(currentWord);
                currentWord = '';
            } else {
                currentWord += token;
            }
        }
        if (currentWord) {
            words.push(currentWord);
        }
        return words.join(' ').replace(/<\/w>/g, ' ').replace(/<\/u>/g, '');
    }

    loadVocab(tokenToIndex) {
        this.tokenToIndex = tokenToIndex;
        this.indexToToken = {};
        for (let [token, index] of Object.entries(tokenToIndex)) {
            this.indexToToken[index] = token;
        }
        this.vocab = new Set(Object.keys(tokenToIndex));
    }

    getVocabSize() {
        return Object.keys(this.tokenToIndex).length;
    }
}

function softmax(arr) 
{
    const maxLogit = Math.max(...arr);
    const scaled = arr.map(logit => Math.exp(logit - maxLogit));
    const total = scaled.reduce((acc, val) => acc + val, 0);
    return scaled.map(val => val / total);
}
function createsourceMask(inputSrc, padIdx) {
    // Step 1: Create a boolean mask
    let mask = inputSrc.map(element => element !== padIdx);

    // Step 2: In JavaScript, we typically don't deal with dimensions in the same way,
    // especially with regular arrays. If you're using a tensor library, you might need
    // to adjust this step to add dimensions.

    // Step 3: Convert boolean mask to integers (0s and 1s)
    let intMask = mask.map(value => value ? 1 : 0);

    // The comparison with 1 in the Python code is redundant since the mask is already in 0s and 1s.
    return intMask;
}
function causalMask(seqLen) {
    let mask = [];
    for (let i = 0; i < seqLen; i++) {
        mask[i] = [];
        for (let j = 0; j < seqLen; j++) {
            mask[i][j] = j <= i ? 1 : 0;
        }
    }
    return mask;
}

function createtargetMask(inputTgt, padIdx, seqLen) {
    // Create the initial mask based on padIdx
    let initialMask = inputTgt.map(element => element !== padIdx ? 1 : 0);

    // Create the causal mask
    let cMask = causalMask(seqLen);

    // Apply the bitwise AND operation
    let finalMask = [];
    for (let i = 0; i < initialMask.length; i++) {
        finalMask[i] = [];
        for (let j = 0; j < seqLen; j++) {
            finalMask[i][j] = initialMask[i] & cMask[i][j];
        }
    }

    return finalMask.flat();
}


async function predict(source, target){
    // prepare inputs. a tensor need its corresponding TypedArray as data
    let source_array = BigInt64Array.from(source.map(item => BigInt(item)));
    let target_array = BigInt64Array.from(target.map(item => BigInt(item)));

    let source_mask_input = createsourceMask(source, pad_token);
    let target_mask_input = createtargetMask(target, pad_token, target.length);
    console.log(target_mask_input);
    source_mask_input = Uint8Array.from(source_mask_input);
    target_mask_input = Uint8Array.from(target_mask_input);

    console.log(target_array);
    console.log(target_mask_input);
    const Tsource = new ort.Tensor('int64', source_array, [1, source_array.length]);
    const Ttarget = new ort.Tensor('int64', target_array, [1, target_array.length]);
    const Tsource_mask = new ort.Tensor('bool', source_mask_input, [1, 1, 1, source_mask_input.length]);
    const Ttarget_mask = new ort.Tensor('bool', target_mask_input, [1, 1, target_array.length, target_array.length]);

    console.log(Tsource);
    console.log(Ttarget);
    console.log(Tsource_mask);
    console.log(Ttarget_mask);

    // prepare feeds. use model input names as keys.
    const feeds = {source: Tsource,
                   target: Ttarget,
                   source_mask: Tsource_mask,
                   target_mask: Ttarget_mask,
                   };
    console.log(feeds);
    // feed inputs and run
    let res = await session.run(feeds)
    let output = res.output.data.slice(-res.output.dims[2]-1, -1)
        
    let index = output.indexOf(Math.max(...output));
    return index;
}
// Function to loop over async operations
async function runAsyncLoop(encodedText, max_gen) {
    let source_input = [...sos_token, ...encodedText, ...eos_token];
    source_input.push(...Array(100 - source_input.length).fill(...pad_token));
    let target = sos_token;
    running = true
    btn.innerHTML = running ? "Wait" : "Start Translate"

    for (let i = 0; (i < max_gen & target[-1]!= eos_token[0]); i++) {
        let result = await predict(source_input, target);
        target.push(result);
        console.log(target)
        console.log(result)
        let ouput = tokenizer.detokenize(target.slice(1, -1));
        console.log(ouput);
        frenchcase.innerText = `Encoded Text: ${ouput}`;
    }
}
const write = () => {
    const inputText = document.getElementById('englishInput').value;
    let encodedText = tokenizer.tokenize(inputText);
    console.log(max_gen);
    console.log(encodedText)
    runAsyncLoop(encodedText, max_gen).then(() => {
        console.log('Finished!');
    }, (reason) => {
        console.log('Error or cancelled:', reason);
    });
};

const writeToggle = () => {
    if (!tokenizer) {
        alert('Dictionary is not loaded yet!');
        return;
    }
    running = !running
    btn.innerHTML = running ? "Wait" : "Start Translate"
    if (running) { write() }
}
btn.addEventListener("click", writeToggle)