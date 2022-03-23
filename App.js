/* eslint-disable prettier/prettier */
import React, { useRef, useState, useEffect } from 'react';
import {
	View,
	StyleSheet,
	Dimensions,
	Pressable,
	Modal,
	Text,
	ActivityIndicator,
} from 'react-native';

import * as tf from '@tensorflow/tfjs';
import {
	startPrediction,
} from './src/helpers/tensor-helper';
import { cropPicture } from './src/helpers/image-helper';
import { bundleResourceIO, decodeJpeg } from '@tensorflow/tfjs-react-native';
import { Camera } from 'expo-camera';
import { cameraWithTensors } from '@tensorflow/tfjs-react-native';

const RESULT_MAPPING = ['Triangle', 'Circle', 'Square'];

const modelJson = require('./src/models/blaze/model.json');
const modelWeights = require('./src/models/blaze/group1-shard1of1.bin');

const TensorCamera = cameraWithTensors(Camera);
const { width } = Dimensions.get('window');
let frameCount = 0;
let makePredictionEneveryNFrame = 50;

export default function App() {
	const cameraRef = useRef();
	const [isProcessing, setIsProcessing] = useState(false);
	const [presentedShape, setPresentedShape] = useState('');

	const handleImageCapture = async (images) => {
		const model = await tf.loadGraphModel(bundleResourceIO(modelJson, modelWeights));
		console.log("handle");
		const loop = async () => {
			if (frameCount % makePredictionEneveryNFrame === 0) {
				const image = images.next().value;
				const x = await tf.cast(image, 'float32')
				await processImagePrediction(model, x);
				tf.dispose(images);
			}

			frameCount += 1;
			frameCount = frameCount % makePredictionEneveryNFrame;

			requestAnimationFrame(loop);
		}
		loop();
	};

	const processImagePrediction = async (model, tensor) => {
		const prediction = await model.predict(tensor.reshape([1, 192, 192, 3]));
		// const preds = prediction.dataSync();
		console.log(prediction);
		// let awareness = "";
		// preds.forEach((pred, i) => {
		// 	//console.log(`x: ${i}, pred: ${pred}`);
		// 	if (pred > 0.9) {
		// 		if (i === 0) {
		// 			awareness = "0";
		// 		}
		// 		if (i === 1) {
		// 			awareness = "10";
		// 		}
		// 		if (i === 2) {
		// 			awareness = "5";
		// 		}
		// 		console.log(`Awareness level ${awareness} Probability : ${pred}`);
		// 	}
		// });
	};

	useEffect(() => {
		(async () => {
			await Camera.requestCameraPermissionsAsync();
			await tf.ready();
		})();
	}, []);

	return (
		<View style={styles.container}>
			<Text style={{
				fontSize: 20,
				color: 'white',
				fontWeight: 'bold',
				zIndex: 99,
				position: 'absolute',
				top: 50,
				left: width / 2 - 50
			}}>
				{presentedShape}
			</Text>
			<TensorCamera
				cameraTextureHeight={1920}
				cameraTextureWidth={1080}
				style={styles.camera}
				type={Camera.Constants.Type.back}
				autoFocus={true}
				whiteBalance={Camera.Constants.WhiteBalance.auto}
				resizeDepth={3}
				resizeHeight={192}
				resizeWidth={192}
				autorender
				onReady={handleImageCapture}
			/>
			<Pressable
				onPress={() => handleImageCapture()}
				style={styles.captureButton}
			/>
		</View>
	);
};

const styles = StyleSheet.create({
	container: {
		flex: 1,
		width: '100%',
		height: '100%',
	},
	camera: {
		width: '100%',
		height: '100%',
	},
	captureButton: {
		position: 'absolute',
		left: Dimensions.get('screen').width / 2 - 50,
		bottom: 40,
		width: 100,
		zIndex: 100,
		height: 100,
		backgroundColor: 'white',
		borderRadius: 50,
	},
	modal: {
		flex: 1,
		width: '100%',
		height: '100%',
		alignItems: 'center',
		justifyContent: 'center',
	},
	modalContent: {
		alignItems: 'center',
		justifyContent: 'center',
		width: 300,
		height: 300,
		borderRadius: 24,
		backgroundColor: 'gray',
	},
	dismissButton: {
		width: 150,
		height: 50,
		marginTop: 60,
		borderRadius: 24,
		color: 'white',
		alignItems: 'center',
		justifyContent: 'center',
		backgroundColor: 'red',
	},
});
