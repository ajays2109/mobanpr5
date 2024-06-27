/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.objectdetection

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.Gravity
import android.view.View
import android.widget.Toast
import androidx.core.content.ContextCompat
import java.util.LinkedList
import kotlin.math.max
import org.tensorflow.lite.task.vision.detector.Detection

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results: List<Detection> = LinkedList<Detection>()
    private var numberPlateResults: List<NumberPlateDetection> = LinkedList<NumberPlateDetection>()
    private var boxPaint = Paint()
    private var textBackgroundPaint = Paint()
    private var textPaint = Paint()

    private var inferenceTimeTaken: Long = 0;

    private var scaleFactor: Float = 1f

    private var bounds = Rect()

    init {
        initPaints()
    }

    fun clear() {
        textPaint.reset()
        textBackgroundPaint.reset()
        boxPaint.reset()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        textBackgroundPaint.color = Color.BLACK
        textBackgroundPaint.style = Paint.Style.FILL
        textBackgroundPaint.textSize = 50f

        textPaint.color = Color.WHITE
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 50f

        boxPaint.color = ContextCompat.getColor(context!!, R.color.bounding_box_color)
        boxPaint.strokeWidth = 8F
        boxPaint.style = Paint.Style.STROKE
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        for (result in results) {
            val boundingBox = result.boundingBox

            val top = boundingBox.top * scaleFactor
            val bottom = boundingBox.bottom * scaleFactor
            val left = boundingBox.left * scaleFactor
            val right = boundingBox.right * scaleFactor

            // Draw bounding box around detected objects
            val drawableRect = RectF(left, top, right, bottom)
            canvas.drawRect(drawableRect, boxPaint)

            // Create text to display alongside detected objects
            val drawableText =
                result.categories[0].label + " " +
                        String.format("%.2f", result.categories[0].score)

            // Draw rect behind display text
            textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
            val textWidth = bounds.width()
            val textHeight = bounds.height()
            canvas.drawRect(
                left,
                top,
                left + textWidth + Companion.BOUNDING_RECT_TEXT_PADDING,
                top + textHeight + Companion.BOUNDING_RECT_TEXT_PADDING,
                textBackgroundPaint
            )

            // Draw text for detected object
            canvas.drawText(drawableText, left, top + bounds.height(), textPaint)
        }

        for (result in numberPlateResults){
            if(result.number?.isEmpty() == true) {
                continue;
            }
            val left = result.left* scaleFactor
            val top = result.top* scaleFactor
            val right = result.right* scaleFactor
            val bottom = result.bottom* scaleFactor
            val number = result.number
            val score = result.score
            // Draw bounding box around detected objects
            val drawableRect = RectF(left, top, right, bottom)
            canvas.drawRect(drawableRect, boxPaint)

            // Create text to display alongside detected objects
            val drawableText =
                number + "(" +
                        String.format("%.2f", score) +")"+
                        " - "+String.format("%.2f", (inferenceTimeTaken/1000.0))+"s"

//            if(score>50) {
//                val toast = Toast.makeText(context, drawableText, Toast.LENGTH_LONG);
//                // Set gravity to the center-right corner of the screen
//                toast.setGravity(Gravity.CENTER_VERTICAL or Gravity.END, 0, 0)
//
//                // Adjust the x offset to create space from the edge of the screen
//                val xOffset = context.resources.displayMetrics.widthPixels / 8 // Adjust as needed
//                toast.setGravity(Gravity.CENTER_VERTICAL or Gravity.END, xOffset, 0)
//                toast.show();
//            }

            // Draw rect behind display text
            textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
            val textWidth = bounds.width()
            val textHeight = bounds.height()
            canvas.drawRect(
                left,
                top,
                left + textWidth + Companion.BOUNDING_RECT_TEXT_PADDING,
                top + textHeight + Companion.BOUNDING_RECT_TEXT_PADDING,
                textBackgroundPaint
            )

            // Draw text for detected object
            canvas.drawText(drawableText, left, top + bounds.height(), textPaint)
            invalidate()
        }
    }

    fun setResults(
      detectionResults: MutableList<Detection>,
      imageHeight: Int,
      imageWidth: Int,
    ) {
        results = detectionResults

        // PreviewView is in FILL_START mode. So we need to scale up the bounding box to match with
        // the size that the captured images will be displayed.
        scaleFactor = max(width * 1f / imageWidth, height * 1f / imageHeight)
    }

    fun setNumberPlateResults(
        detectedNumberPlateResults: List<NumberPlateDetection>,
        imageHeight: Int,
        imageWidth: Int,
        inferenceTime: Long,
    ){
        inferenceTimeTaken = inferenceTime;
        results = LinkedList();
        numberPlateResults = detectedNumberPlateResults
        scaleFactor = max(width * 1f / imageWidth, height * 1f / imageHeight)
        invalidate()
    }

    companion object {
        private const val BOUNDING_RECT_TEXT_PADDING = 8
    }
}
