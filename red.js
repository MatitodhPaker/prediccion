let entradas=[
    [100,6,75],
    [50,4,55],
    [25,3,35],
    [15,1,15]
];
let resultado=[
    [1],
    [0.5],
    [0.25],
    [0]
];
let inputs=tf.tensor2d(entradas);
let outputs=tf.tensor2d(resultado);
async function crearModelo() {
    const modelo = tf.sequential();
    const hiden = tf.layers.dense({ inputShape: [3], units: 3,activation: 'tanh'});
    modelo.add(hiden);
    const output = tf.layers.dense({ inputShape: [3], units: 1,activation: 'tanh'});
    modelo.add(output);
    modelo.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
    })
    const configTrain = {
        epochs: 1000
    };

    const h = await modelo.fit(inputs, outputs, configTrain);
    console.log(h);
    let prediccion = modelo.predict(tf.tensor2d(entradas));
    prediccion.print();
    outputs.print();
}
crearModelo();