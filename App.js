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
import Canvas from 'react-native-canvas';


import * as tf from '@tensorflow/tfjs';
import {
	startPrediction,
} from './src/helpers/tensor-helper';
import { cropPicture } from './src/helpers/image-helper';
import { bundleResourceIO, decodeJpeg } from '@tensorflow/tfjs-react-native';
import { Camera } from 'expo-camera';
import { cameraWithTensors } from '@tensorflow/tfjs-react-native';
import Svg, { Rect } from 'react-native-svg';


const modelJson = require('./src/models/blaze/model.json');
const modelWeights = require('./src/models/blaze/group1-shard1of1.bin');

const TensorCamera = cameraWithTensors(Camera);
const { width, height } = Dimensions.get('window');
let frameCount = 0;
let makePredictionEneveryNFrame = 50;

export default function App() {
	const [face, setFace] = useState();

	const handleImageCapture = async (images) => {
		const model = await tf.loadGraphModel(bundleResourceIO(modelJson, modelWeights));
		const loop = async () => {
			if (frameCount % makePredictionEneveryNFrame === 0) {
				const image = images.next().value;
				const tensor = await tf.cast(image, 'float32')
				await processImagePrediction(model, tensor.reshape([1, 192, 192, 3]));
				tf.dispose(images);
			}

			frameCount += 1;
			frameCount = frameCount % makePredictionEneveryNFrame;

			requestAnimationFrame(loop);
		}
		loop();
	};

	const processImagePrediction = async (model, tensor) => {
		const output = model.execute(tensor);
		const [boxes, scores, classes] = output;
		const boxes_data = boxes.dataSync();
		let [x1, y1, x2, y2] = boxes_data.slice(0, 4);
		const width = x2 - x1;
		const height = y2 - y1;
		setFace({
			x: x1,
			y: y1,
			width,
			height,
		})
	};

	const renderBoundings = () => {
		const { x, y, width, height } = face;
		return (
			<Svg
				height="100%"
				width="100%"
				viewBox={`0 0 ${width} ${height}`}
			>
				<Rect
					x={x1}
					y={y1}
					width={width}
					height={height}
					stroke="red"
					strokeWidth="2"
				/>
			</Svg>
		);
	}

	useEffect(() => {
		(async () => {
			await Camera.requestCameraPermissionsAsync();
			await tf.ready();
		})();
	}, []);

	return (
		<View style={styles.container}>
			<TensorCamera
				cameraTextureHeight={1920}
				cameraTextureWidth={1080}
				style={styles.camera}
				type={Camera.Constants.Type.front}
				autoFocus={true}
				whiteBalance={Camera.Constants.WhiteBalance.auto}
				resizeDepth={3}
				resizeHeight={192}
				resizeWidth={192}
				autorender
				onReady={handleImageCapture}
			/>
			<View style={{
				position: 'absolute',
				width: '100%',
				height: '100%',
				zIndex: 999
			}}>
				{renderBoundings()}
			</View>

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
