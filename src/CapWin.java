import javax.swing.JFrame;
import javax.swing.JPanel;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import pers.season.vml.ar.CameraData;
import pers.season.vml.ar.Engine3D;
import pers.season.vml.ar.TemplateDetector;
import pers.season.vml.ar.MotionFilter;
import pers.season.vml.util.ImUtils;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JLabel;

public class CapWin extends JFrame {
	static {
		System.loadLibrary("lib/opencv_java320_x64");
	}

	public static void main(String[] args) {
		CapWin win = new CapWin();
		win.setVisible(true);

		Mat template = Imgcodecs.imread("./target.jpg");
		ImUtils.imshow(template);
		TemplateDetector td = new TemplateDetector();
		td.setTemplate(template);

		VideoCapture vc = new VideoCapture();
		vc.open(0);
		Mat cam = new Mat();
		Mat vcam = new Mat();
		while (true) {
			vc.read(cam);
			cam.copyTo(vcam);
			Mat homo = td.findHomo(cam, true);
			if (homo != null) {
				Mat quad = td.getQuadFromHomo(homo);

				for (int i = 0; i < quad.total(); i++) {
					Imgproc.line(vcam, new Point(quad.get(i, 0)[0], quad.get(i, 0)[1]),
							new Point(quad.get((int) ((i + 1) % quad.total()), 0)[0],
									quad.get((int) ((i + 1) % quad.total()), 0)[1]),
							new Scalar(0, 255, 0), 3);
				}

				Mat rvec = new Mat(), tvec = new Mat();
				td.solvePnp(homo, CameraData.MY_CAMERA, rvec, tvec);
			}
			if (win.captureButtonClicked) {
				win.captureButtonClicked = false;
				Imgcodecs.imwrite("./" + System.currentTimeMillis() + ".jpg", cam);
				Imgcodecs.imwrite("./" + System.currentTimeMillis() + "_line.jpg", vcam);
			}
			ImUtils.imdraw(win.lblCamera, vcam, 1f);
		}

	}

	private JButton btnCapture;
	private JLabel lblCamera;
	private boolean captureButtonClicked = false;

	public CapWin() {
		getContentPane().setLayout(null);

		btnCapture = new JButton("Capture");
		btnCapture.setBounds(320-50, 490, 100, 30);
		btnCapture.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent arg0) {
				captureButtonClicked = true;
			}
		});
		getContentPane().add(btnCapture);

		lblCamera = new JLabel("Camera");
		lblCamera.setBounds(0, 0, 640, 480);
		getContentPane().add(lblCamera);

		this.setSize(new Dimension(640 + 50, 600 ));
	}
}
