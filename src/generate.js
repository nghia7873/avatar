import * as tf from '@tensorflow/tfjs';
import * as tfc from '@tensorflow/tfjs-converter';
import Pica from 'pica';

const pica = new Pica();

window.tf = tf;
window.tfc = tfc;
window.progress = 0;
window.bytesUsed = 0;

tf.enableProdMode();

let start;

const MODEL_URL = window.location.href + 'model_full/model.json';

function mirrorPadFunc(input, pad_arr) {
    return tf.tidy(() => {
        for (let i = 0; i < 4; i++) {
            if (pad_arr[i][0] !== 0 || pad_arr[i][1] !== 0) {
                let slice_size = [-1, -1, -1, -1];
                slice_size[i] = pad_arr[i][0];
                let slice_begin = [0, 0, 0, 0];

                let padding_left = input.slice(slice_begin, slice_size);

                slice_size = [-1, -1, -1, -1];
                slice_size[i] = pad_arr[i][1];
                slice_begin = [0, 0, 0, 0];
                slice_begin[i] = input.shape[i] - pad_arr[i][1];

                let padding_right = input.slice(slice_begin, slice_size);

                input = tf.concat([padding_left, input, padding_right], i);
            }

            if (pad_arr[i][0] > 1 || pad_arr[i][1] > 1) {
                throw new Error("Only input with no more than length one in padding is supported. We have: " + JSON.stringify(pad_arr));
            }
        }
        return input;
    });
}

// For debugging purpose:
window.mirrorPadFunc = mirrorPadFunc;

const progressesList = [0.00023367749587460492, 0.054088046653978504, 0.1804816724673639, 0.18052037621199904, 0.2528568019649621, 0.37458444400475477, 0.39315031021211105, 0.39319017797911254, 0.4444196766347441, 0.5207431700988491, 0.550593651422125, 0.5542242372745627, 0.5605664132978859, 0.5806242652109398, 0.5927784050567816, 0.5962346785553008, 0.5981026434950807, 0.5989430676647844, 0.6435568450337933, 0.6676838282371483, 0.6684442258671517, 0.7463103400111626, 0.9019785470675509, 0.95];
let num_called = 0;

const mirrorPad = async (node) => {
    let progress = 0.9 * (performance.now() - start)/(15463.61999999499);

    /* progressesList.push(progress);
    console.log(progressesList); */

    if (num_called >= progressesList.length) {
        progress = 0.95;
    } else {
        progress = progressesList[num_called];
    }
    num_called += 1;

    window.progress = progress;

    let memoryInfo = tf.memory();
    console.log("Memory Info:", memoryInfo);
    window.bytesUsed = memoryInfo.numBytes;

    // Use normal pad (not mirror pad):
    // return tf.pad(node.inputs[0], node.inputs[1].arraySync(), 0);

    await tf.nextFrame();

    if (node.attrs.mode !== "reflect") {
        throw new Error("Only reflect mode is supported. Mode: " + node.attrs.mode);
    }
    let pad_tensor = node.inputs[1];
    // node.inputs[1].print();
    if (node.inputs[0].shape.length === 4) {
        let pad_arr = await pad_tensor.array();
        let input = node.inputs[0];
        return mirrorPadFunc(input, pad_arr);
    } else {
        throw new Error("Only input of rank 4 is supported. We have: " + JSON.stringify(pad_tensor.arraySync()));
    }
};

tfc.registerOp('MirrorPad', mirrorPad);

const generate = async (model, scaled_size, img, output) => {
    const img_new = await resizeImage(img)

    const img_tensor = tf.browser.fromPixels(img_new).toFloat();
    let scaled_img_tensor = tf.image.resizeBilinear(img_tensor, scaled_size).div(255).expandDims(0);

    let start = performance.now();
    let generated = await model.executeAsync({'test': scaled_img_tensor});
    let end = performance.now();

    tf.browser.toPixels((generated.squeeze().add(1)).div(2), output);
    generated.dispose();
    scaled_img_tensor.dispose();

    return end - start;
}

let preHeat = () => {
    let model_load_start = performance.now();
    tfc.loadGraphModel(MODEL_URL).then((model) => {
        console.log("Model Loaded");
        let model_load_end = performance.now();
        console.log(`Took ${(model_load_end - model_load_start)/1000} s to load the model`);
        model.dispose();
    });
}

const loadModel = async () => {
    const model = await tf.loadGraphModel(MODEL_URL);
    return model;
}
const resizeImage = async (img) => {
    const resizedCanvas = document.createElement('canvas');
    resizedCanvas.width = 256;
    resizedCanvas.height = 256;
    return await pica.resize(img, resizedCanvas);
}

const generateImage = async (resize, fp16, img_id, canvas_id, tf_backend) => {
    tf.ENV.set('WEBGL_FORCE_F16_TEXTURES', true);

    tf.setBackend(tf_backend);

    let scaled_size;

    if (resize === "s") {
        scaled_size = [100, 100];
    } else if (resize === "m") {
        scaled_size = [250, 250];
    } else if (resize === "l") {
        scaled_size = [500, 500];
    }

    const model = await loadModel();
    const processing_time = await generate(model, scaled_size, document.getElementById(img_id), document.getElementById(canvas_id));
    model.dispose();
    tf.disposeVariables();

    window.progress = 1.0;

    return processing_time;
};

export {preHeat, generateImage};
