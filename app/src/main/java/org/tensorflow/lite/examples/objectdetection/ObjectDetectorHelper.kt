/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.tensorflow.lite.examples.objectdetection

import android.content.Context
import android.content.res.AssetFileDescriptor
import android.content.res.AssetManager
import android.content.res.Configuration
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.Detection
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.LinkedList

class ObjectDetectorHelper(
  var threshold: Float = 0.5f,
  var numThreads: Int = 2,
  var maxResults: Int = 3,
  var currentDelegate: Int = 0,
  var currentModel: Int = 4,
  val context: Context,
  val objectDetectorListener: DetectorListener?
) {

    // For this example this needs to be a var so it can be reset on changes. If the ObjectDetector
    // will not change, a lazy val would be preferable.
    private var objectDetector: ObjectDetector? = null
    private var numberPlateDetector: Interpreter? = null
    private var textRecognizer: TextRecognizer? = null
    private val minConf: Float = 0.5f // Minimum confidence threshold
    init {
        setupObjectDetector()
    }

    fun clearObjectDetector() {
        numberPlateDetector = null
        objectDetector = null
        textRecognizer = null
    }

    // Initialize the object detector using current settings on the
    // thread that is using it. CPU and NNAPI delegates can be used with detectors
    // that are created on the main thread and used on a background thread, but
    // the GPU delegate needs to be used on the thread that initialized the detector
    fun setupObjectDetector() {

        if( currentModel == MODEL_NUMBERPLATE){
            try {
                numberPlateDetector = Interpreter(loadNumberPlateModelFile())
                textRecognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
            } catch (e: IllegalStateException) {
            objectDetectorListener?.onError(
                    "Object detector failed to initialize. See error logs for details"
                )
                Log.e("test", "TFLite failed to load model with error: " + e.message)
            }
        }
        else {
            // Create the base options for the detector using specifies max results and score threshold
            val optionsBuilder =
                ObjectDetector.ObjectDetectorOptions.builder()
                    .setScoreThreshold(threshold)
                    .setMaxResults(maxResults)

            // Set general detection options, including number of used threads
            val baseOptionsBuilder = BaseOptions.builder().setNumThreads(numThreads)

            // Use the specified hardware for running the model. Default to CPU
            when (currentDelegate) {
                DELEGATE_CPU -> {
                    // Default
                }

                DELEGATE_GPU -> {
                    if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                        baseOptionsBuilder.useGpu()
                    } else {
                        objectDetectorListener?.onError("GPU is not supported on this device")
                    }
                }

                DELEGATE_NNAPI -> {
                    baseOptionsBuilder.useNnapi()
                }
            }

            optionsBuilder.setBaseOptions(baseOptionsBuilder.build())

            val modelName =
                when (currentModel) {
                    MODEL_MOBILENETV1 -> "mobilenetv1.tflite"
                    MODEL_EFFICIENTDETV0 -> "efficientdet-lite0.tflite"
                    MODEL_EFFICIENTDETV1 -> "efficientdet-lite1.tflite"
                    MODEL_EFFICIENTDETV2 -> "efficientdet-lite2.tflite"
                    MODEL_NUMBERPLATE -> "detect.tflite"
                    else -> "mobilenetv1.tflite"
                }

            try {
                objectDetector =
                    ObjectDetector.createFromFileAndOptions(
                        context,
                        modelName,
                        optionsBuilder.build()
                    )
            } catch (e: IllegalStateException) {
                objectDetectorListener?.onError(
                    "Object detector failed to initialize. See error logs for details"
                )
                Log.e("Test", "TFLite failed to load model with error: " + e.message)
            }
        }
    }

    fun detect(image: Bitmap, imageRotation: Int) {
        if (objectDetector == null && numberPlateDetector == null) {
            setupObjectDetector()
        }

        // Inference time is the difference between the system time at the start and finish of the
        // process
        var inferenceTime = SystemClock.uptimeMillis()

        // Create preprocessor for the image.
        // See https://www.tensorflow.org/lite/inference_with_metadata/
        //            lite_support#imageprocessor_architecture
        if(objectDetector != null) {
            val imageProcessor =
                ImageProcessor.Builder()
                    .add(Rot90Op(-imageRotation / 90))
                    .build()

            // Preprocess the image and convert it into a TensorImage for detection.
            val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))

            val results = objectDetector?.detect(tensorImage)
            inferenceTime = SystemClock.uptimeMillis() - inferenceTime
            objectDetectorListener?.onResults(
                results,
                inferenceTime,
                tensorImage.height,
                tensorImage.width
            )
        }
        if(numberPlateDetector != null){
            detectNumberPlate(image, imageRotation)
        }
    }

    private fun detectNumberPlate(image: Bitmap, imageRotation: Int){
        var rotatedBitmap = image;
        if(rotatedBitmap.width > rotatedBitmap.height) {
            rotatedBitmap = rotateBitmap(image, 90)
        }
        val scaleFactor = 0.5f
        rotatedBitmap = Bitmap.createScaledBitmap(rotatedBitmap, (rotatedBitmap.width * scaleFactor).toInt(), (rotatedBitmap.height * scaleFactor).toInt(), true)

        // Inference time is the difference between the system time at the start and finish of the process
        var inferenceTime = SystemClock.uptimeMillis()

        val inputShape = numberPlateDetector!!.getInputTensor(0).shape()

        // Resize bitmap to match input shape
        val resizedBitmap = Bitmap.createScaledBitmap(rotatedBitmap, inputShape[1], inputShape[2], true)

        // Prepare input tensor
        val inputImageBuffer = ByteBuffer.allocateDirect(inputShape[1] * inputShape[2] * 3 * 4).apply {
            order(ByteOrder.nativeOrder())
            rewind()
        }

        // Convert bitmap to ByteBuffer and normalize if needed
        val intValues = IntArray(inputShape[1] * inputShape[2])
        resizedBitmap.getPixels(intValues, 0, resizedBitmap.width, 0, 0, resizedBitmap.width, resizedBitmap.height)
        var pixel = 0
        for (i in 0 until inputShape[1]) {
            for (j in 0 until inputShape[2]) {
                val value = intValues[pixel++]

                // Normalize the pixel values if the model requires it
                val normalizedR = ((value shr 16 and 0xFF) - 127.5f) / 127.5f
                val normalizedG = ((value shr 8 and 0xFF) - 127.5f) / 127.5f
                val normalizedB = ((value and 0xFF) - 127.5f) / 127.5f

                inputImageBuffer.putFloat(normalizedR)
                inputImageBuffer.putFloat(normalizedG)
                inputImageBuffer.putFloat(normalizedB)
            }
        }

        val outputBufferScores = TensorBuffer.createFixedSize(numberPlateDetector!!.getOutputTensor(0).shape(), numberPlateDetector!!.getOutputTensor(0).dataType())
        val outputBufferBoxes = TensorBuffer.createFixedSize(numberPlateDetector!!.getOutputTensor(1).shape(), numberPlateDetector!!.getOutputTensor(1).dataType())
        val outputBufferClasses = TensorBuffer.createFixedSize(numberPlateDetector!!.getOutputTensor(3).shape(), numberPlateDetector!!.getOutputTensor(3).dataType())

        try {
            numberPlateDetector?.runForMultipleInputsOutputs(arrayOf(inputImageBuffer), mapOf(0 to outputBufferScores.buffer, 1 to outputBufferBoxes.buffer, 3 to outputBufferClasses.buffer))
        } catch (e: Exception) {
            Log.e(TAG, "Failed to run inference with error: " + e.message)
        }
        val scores = outputBufferScores.floatArray
        val boxes = outputBufferBoxes.floatArray
        val classes = outputBufferClasses.floatArray

        // Log the scores for debugging
        //Log.e(TAG, "Scores: ${scores.joinToString(", ")}")

        // Assuming labels are loaded into a list called 'labels'
//        var i=0;
        val canvas = Canvas(rotatedBitmap)

        for(i in scores.indices){
            val confidence = scores[i]
            if (confidence >= minConf) {
                val classIndex = classes[i].toInt()
                val ymin = (boxes[i * 4] * rotatedBitmap.height).toInt()
                val xmin = (boxes[i * 4 + 1] * rotatedBitmap.width).toInt()
                val ymax = (boxes[i * 4 + 2] * rotatedBitmap.height).toInt()
                val xmax = (boxes[i * 4 + 3] * rotatedBitmap.width).toInt()

                // Ensure bounding box is within image boundaries
                val boundedXmin = xmin.coerceAtLeast(0)
                val boundedYmin = ymin.coerceAtLeast(0)
                val boundedXmax = xmax.coerceAtMost(rotatedBitmap.width)
                val boundedYmax = ymax.coerceAtMost(rotatedBitmap.height)

                // Calculate width and height of the bounding box
                val boxWidth = boundedXmax - boundedXmin
                val boxHeight = boundedYmax - boundedYmin

                // Ensure width and height are positive
                if (boxWidth > 32 && boxHeight > 32) {
                    //Log.e(TAG, "class index = ${classIndex} | confidence = ${confidence * 100}")

                    // Create a bitmap for the detected region
                    val detectedRegionBitmap = Bitmap.createBitmap(rotatedBitmap, boundedXmin, boundedYmin, boxWidth, boxHeight)

                    val grayscaleBitmap = toGrayscale(detectedRegionBitmap)
                    val contrastBitmap = increaseContrast(grayscaleBitmap, 2.5f)

                    val ocrImage = InputImage.fromBitmap(contrastBitmap, 0)
                    val result = textRecognizer?.let { Tasks.await(it.process(ocrImage)) }
                    //val result = "License Plate";
//                    if (result != null) {
//                        Log.i(TAG ,"Detected Plate - ${result.text}")
//                    }
    //                    Toast.makeText(context,result?.text,Toast.LENGTH_LONG).show()
                    // Draw bounding box
    //                val paint1 = Paint().apply {
    //                    color = Color.RED
    //                    style = Paint.Style.STROKE
    //                    strokeWidth = 4.0f // Adjust thickness as needed
    //                }
    //
    //                canvas.drawRect(
    //                    RectF(boundedXmin.toFloat(), boundedYmin.toFloat(), boundedXmax.toFloat(), boundedYmax.toFloat()),
    //                    paint1
    //                )
    //
                    //Log.e(TAG, "Detected: ${classIndex} with confidence ${confidence * 100}%")

                    // Optionally save or use the detectedRegionBitmap
                    // For example, you could save it to a file or display it in an ImageView
                    if(result!= null) {
//                        val numberPlatePattern = Regex("[A-Z0-9]{1,3}-[A-Z0-9]{1,4}|[A-Z0-9]{1,7}")
                        val filteredText = filterText(result);
                        var textToDisplay = result.text;
                        if(filteredText != null)
                            textToDisplay = filteredText;
                        val numberPlateResults: List<NumberPlateDetection> = listOf(
                            NumberPlateDetection(
                                boundedXmin.toFloat(),
                                boundedYmin.toFloat(),
                                boundedXmax.toFloat(),
                                boundedYmax.toFloat(),
                                textToDisplay,
                                confidence*100
                            )
                        )
//                        val numberPlateResults: List<NumberPlateDetection> = listOf(
//                            NumberPlateDetection(
//                                boundedXmin.toFloat(),
//                                boundedYmin.toFloat(),
//                                boundedXmax.toFloat(),
//                                boundedYmax.toFloat(),
//                                result,
//                                confidence*100
//                            )
//                        )

                        inferenceTime = SystemClock.uptimeMillis() - inferenceTime

                        objectDetectorListener?.onNumberPlateResults(
                            numberPlateResults,
                            inferenceTime,
                            rotatedBitmap.height,
                            rotatedBitmap.width
                        )
                    }
                    else{

                        val numberPlateResults: List<NumberPlateDetection> = LinkedList()

                        inferenceTime = SystemClock.uptimeMillis() - inferenceTime

                        objectDetectorListener?.onNumberPlateResults(
                            numberPlateResults,
                            inferenceTime,
                            0,
                            0
                        )
                    }
                }
            }
        }
    }
    fun rotateBitmap(source: Bitmap, rotation: Int): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(rotation.toFloat())
        return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
    }

    fun filterText(result: Text): String? {
        val blocks = result.textBlocks
        val largestBlock = blocks.maxByOrNull { it.boundingBox?.height() ?: 0 }

        if (largestBlock != null) {
            val filteredText = largestBlock.text.replace("[^A-Za-z0-9]".toRegex(), "")
            return filteredText;
        }
        return null;
    }

    fun toGrayscale(bitmap: Bitmap): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        val grayscaleBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(grayscaleBitmap)
        val paint = Paint()
        val colorMatrix = ColorMatrix()
        colorMatrix.setSaturation(0f)
        val colorMatrixFilter = ColorMatrixColorFilter(colorMatrix)
        paint.colorFilter = colorMatrixFilter
        canvas.drawBitmap(bitmap, 0f, 0f, paint)
        return grayscaleBitmap
    }

    fun increaseContrast(bitmap: Bitmap, value: Float): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        val contrastBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(contrastBitmap)
        val paint = Paint()
        val contrast = value
        val scale = contrast + 1.0f
        val translate = (-.5f * scale + .5f) * 255.0f
        val colorMatrix = ColorMatrix(floatArrayOf(
            scale, 0f, 0f, 0f, translate,
            0f, scale, 0f, 0f, translate,
            0f, 0f, scale, 0f, translate,
            0f, 0f, 0f, 1f, 0f
        ))
        val colorMatrixFilter = ColorMatrixColorFilter(colorMatrix)
        paint.colorFilter = colorMatrixFilter
        canvas.drawBitmap(bitmap, 0f, 0f, paint)
        return contrastBitmap
    }

    private fun loadNumberPlateModelFile(): MappedByteBuffer {
        val assetManager: AssetManager = context.assets
        val assetFileDescriptor: AssetFileDescriptor = assetManager.openFd("detect.tflite")
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel: FileChannel = fileInputStream.channel
        val startOffset: Long = assetFileDescriptor.startOffset
        val declaredLength: Long = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    interface DetectorListener {
        fun onError(error: String)
        fun onResults(
          results: MutableList<Detection>?,
          inferenceTime: Long,
          imageHeight: Int,
          imageWidth: Int
        )
        fun onNumberPlateResults(
            numberPlateResults:List<NumberPlateDetection>?,
            inferenceTime: Long,
            imageHeight: Int,
            imageWidth: Int
        )
    }

    companion object {
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DELEGATE_NNAPI = 2
        const val MODEL_MOBILENETV1 = 0
        const val MODEL_EFFICIENTDETV0 = 1
        const val MODEL_EFFICIENTDETV1 = 2
        const val MODEL_EFFICIENTDETV2 = 3
        const val MODEL_NUMBERPLATE = 4
        private const val TAG = "ObjectDetectorHelper"

    }
}
class NumberPlateDetection(
    val left: Float, val top: Float,
    val right: Float, val bottom: Float,
    val number: String?,
    val score: Float
){

}
