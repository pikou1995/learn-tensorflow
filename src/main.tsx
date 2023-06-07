import * as tf from "@tensorflow/tfjs";
// Add the WebGPU backend to the global backend registry.
import "@tensorflow/tfjs-backend-webgpu";
// Set the backend to WebGPU and wait for the module to be ready.
tf.setBackend("webgpu").then(() => main());

async function main() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [1], units: 1 }));
  model.compile({
    optimizer: "sgd",
    loss: "meanSquaredError",
  });
  const xs = [-1, 0, 1, 2, 3, 4];
  // y = 2x - 1
  function targetF(x: number) {
    return 2 * x - 1;
  }
  const ys = xs.map(targetF);
  console.time("fit");
  await model.fit(tf.tensor1d(xs), tf.tensor1d(ys), {
    epochs: 500,
    callbacks: {
      onYield: console.log,
    },
  });
  console.timeEnd("fit");
  const resultTensor = model.predict(tf.tensor1d([10])) as tf.Tensor;
  const result = await resultTensor.data();
  console.log(result[0], targetF(10));
}
