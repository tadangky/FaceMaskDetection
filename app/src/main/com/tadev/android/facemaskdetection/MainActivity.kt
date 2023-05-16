package com.tadev.android.facemaskdetection

import android.Manifest.permission.CAMERA
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.Image
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.DisplayMetrics
import android.util.Log
import android.widget.Toast
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
//import com.google.android.gms.tflite.acceleration.Model
import com.google.common.util.concurrent.ListenableFuture
import com.tadev.android.facemaskdetection.ml.FackMaskDetection
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.support.model.Model.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

class MainActivity : AppCompatActivity() {
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private val lensFacing: Int = CameraSelector.LENS_FACING_FRONT
    private var camera: Camera? = null

    private lateinit var cameraExcutor: ExecutorService
    private lateinit var faceMaskDetection: FackMaskDetection
    private lateinit var yuvToRgbConverter: YuvToRgbConverter
    private lateinit var bitmapBuffer: Bitmap
    private lateinit var rotationMatrix: Matrix

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        setupML()
        setupCameraThread()

        if (!allPermissionsGranted) {
            requireCameraPermission()
        } else {
            setupCamera()
        }
    }

    private fun setupCamera() {
        val cameraProviderFuture: ListenableFuture<ProcessCameraProvider> =
            ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener(
            {
                cameraProvider = cameraProviderFuture.get()
                setupCameraUseCases()
            }, ContextCompat.getMainExecutor(this)
        )
    }

    private fun requireCameraPermission() {
        ActivityCompat.requestPermissions(
            this, REQUIRED_PERMISSIONS, 1
        )

    }

    private fun setupCameraUseCases() {
        try {
            val cameraSelector: CameraSelector =
                CameraSelector.Builder().requireLensFacing(lensFacing).build()

            val metrics: DisplayMetrics =
                DisplayMetrics().also { previewView.display.getRealMetrics(it) }
            val rotation: Int = previewView.display.rotation
            val screenAspectRatio: Int = aspectRatio(metrics.widthPixels, metrics.heightPixels)
            preview = Preview.Builder()
                .setTargetAspectRatio(screenAspectRatio)
                .setTargetRotation(rotation)
                .build()

            imageAnalyzer = ImageAnalysis.Builder()
                .setTargetAspectRatio(screenAspectRatio)
                .setTargetRotation(rotation)
                .build()
                .also {
                    it.setAnalyzer(cameraExcutor) { imageProxy: ImageProxy ->
                        val bitmap = imageProxy.toBitmap()
                        bitmap?.let { it1 ->
                            setupMLOutput(it1)
                            imageProxy.close()
                        }
                    }
                }
            cameraProvider?.unbindAll()

            camera = cameraProvider?.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalyzer
            )
            preview?.setSurfaceProvider(previewView.surfaceProvider)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun setupMLOutput(bitmap: Bitmap) {
        val tensorImage: TensorImage = TensorImage.fromBitmap(bitmap)
        val result: FackMaskDetection.Outputs = faceMaskDetection.process(tensorImage)

        val output: List<Category> = result.probabilityAsCategoryList.apply {
            sortByDescending { res -> res.score }
        }

        GlobalScope.launch(Dispatchers.Main) {
            output.firstOrNull()?.let { category ->
                Log.e("TATA", "category.label ${category.label}")
                status.text = category.label
            }
        }
    }

    private fun aspectRatio(widthPixels: Int, heightPixels: Int): Int {
        val previewRatio: Double =
            max(widthPixels, heightPixels).toDouble() / min(widthPixels, heightPixels)
        if (abs(previewRatio - RATIO_4_3_VALUE) <= abs(previewRatio - RATIO_16_9_VALUE)) {
            return AspectRatio.RATIO_4_3
        }
        return AspectRatio.RATIO_16_9
    }


    private fun setupCameraThread() {
        cameraExcutor = Executors.newCachedThreadPool()

    }

    private fun setupML() {
        yuvToRgbConverter = YuvToRgbConverter(this)

        val options = Options.Builder().setDevice(Device.GPU).setNumThreads(5).build()
        faceMaskDetection = FackMaskDetection.newInstance(applicationContext, options)
    }

    private fun grantedCameraPermission(requestCode: Int) {
        if (requestCode == 1) {
            if (allPermissionsGranted) {
                setupCamera()
            } else {
                Toast.makeText(this, "Permission!!!", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    private val allPermissionsGranted: Boolean
        get() {
            return REQUIRED_PERMISSIONS.all {
                ContextCompat.checkSelfPermission(
                    baseContext,
                    it
                ) == PackageManager.PERMISSION_GRANTED
            }
        }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        grantedCameraPermission(1)
    }


    @SuppressLint("UnsafeOptInUsageError")
    private fun ImageProxy.toBitmap(): Bitmap? {
        val image: Image = this.image ?: return null
        if (!::bitmapBuffer.isInitialized) {
            rotationMatrix = Matrix()
            rotationMatrix.postRotate(this.imageInfo.rotationDegrees.toFloat())
            bitmapBuffer = Bitmap.createBitmap(this.width, this.height, Bitmap.Config.ARGB_8888)
        }

        yuvToRgbConverter.yuvToRgb(image, bitmapBuffer)

        return Bitmap.createBitmap(
            bitmapBuffer,
            0,
            0,
            bitmapBuffer.width,
            bitmapBuffer.height,
            rotationMatrix,
            false
        )
    }

    companion object {
        private val REQUIRED_PERMISSIONS: Array<String> = arrayOf(CAMERA)
        private const val RATIO_4_3_VALUE: Double = 4.0 / 3.0
        private const val RATIO_16_9_VALUE: Double = 16.0 / 9.0
    }
}
