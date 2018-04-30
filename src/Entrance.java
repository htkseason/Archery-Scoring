import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;

import javax.swing.JFrame;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import pers.season.vml.ar.ArUcoDetector;
import pers.season.vml.ar.CameraData;
import pers.season.vml.ar.Engine3D;
import pers.season.vml.ar.TemplateDetector;
import pers.season.vml.ar.MotionFilter;
import pers.season.vml.ar.MyArUco;
import pers.season.vml.util.*;

public final class Entrance {
	static ArUcoDetector ard = new ArUcoDetector(40, 50, 20);
	static {
		System.loadLibrary("lib/opencv_java320_x64");
	}

	static Mat getPlaneVector(Point center, Mat K, Mat homo, Mat lines) {
		// get camera location
		Mat H = new Mat();
		Core.gemm(K.inv(), homo, 1, new Mat(), 0, H);
		double lambda = (Core.norm(H.col(0)) + Core.norm(H.col(1))) * 0.5;
		Core.divide(H, new Scalar(lambda), H);
		Mat t = H.col(2).clone();
		Mat R = H.clone();
		R.col(0).cross(R.col(1)).copyTo(R.col(2));
		Mat u = new Mat(), vt = new Mat();
		Core.SVDecomp(R, new Mat(), u, vt, Core.SVD_FULL_UV);
		Core.gemm(u, vt, 1, new Mat(), 0, R);
		
		Mat camLoc = new MatOfPoint3f();
		Core.multiply(R, new Scalar(-1), R);
		Core.gemm(R.t(), t, 1, new Mat(), 0, camLoc);
		camLoc.put(2, 0, -camLoc.get(2, 0)[0]);
		camLoc = camLoc.reshape(3);
		camLoc.convertTo(camLoc, CvType.CV_32FC3);
		
		// get line abc through center
		double[] abc = new double[3];
		for (int i = 0; i < lines.rows(); i++) {
			double[] tabc = getLineABC(lines.row(i));
			abc[0] += tabc[0];
			abc[1] += tabc[1];
		}
		abc[0] /= lines.rows();
		abc[1] /= lines.rows();
		abc[2] = -abc[0] * center.x - abc[1] * center.y;

		// calc plane vector
		Mat p = new MatOfPoint3f(new Point3(0, -abc[2] / abc[1], 0));
		Mat centerP = new MatOfPoint3f(new Point3(center.x, center.y, 0));
		Mat v1 = new Mat(), v2 = new Mat();
		Core.subtract(p, camLoc, v1);
		Core.subtract(centerP, camLoc, v2);
		return v1.cross(v2);
	}

	static Mat getInAngle(Point center, Mat homo1, Mat homo2, Mat lines1, Mat lines2) {
		Mat n1 = getPlaneVector(center, CameraData.MY_CAMERA, homo1, lines1);
		Mat n2 = getPlaneVector(center, CameraData.MY_CAMERA, homo2, lines2);
		Mat v = n1.cross(n2);
		Core.normalize(v, v);
		return v;
	}

	static void findLines(Mat camx_0, Mat camx_1, Mat template, Mat outLines, Mat outHomo) {
		getHomo(camx_0, template).copyTo(outHomo);
		Mat homo = outHomo;
		Mat bg = new Mat();
		Imgproc.warpPerspective(camx_0, bg, homo, template.size(), Imgproc.WARP_INVERSE_MAP | Imgproc.INTER_CUBIC);
		Mat bow = new Mat();
		Imgproc.warpPerspective(camx_1, bow, homo, template.size(), Imgproc.WARP_INVERSE_MAP | Imgproc.INTER_CUBIC);
		Mat sub = new Mat();
		Core.absdiff(bg, bow, sub);

		TermCriteria tc = new TermCriteria(TermCriteria.MAX_ITER | TermCriteria.EPS, 100, 0.01);
		Mat centers = new Mat();
		Mat labels = new Mat();
		Mat kmsub = new Mat();
		sub.reshape(1, (int) sub.total()).convertTo(kmsub, CvType.CV_32F);
		Core.kmeans(kmsub, 4, labels, tc, 10, Core.KMEANS_PP_CENTERS, centers);
		labels = labels.reshape(1, sub.rows());
		double[] centersArr = new double[centers.rows()];
		for (int i = 0; i < centers.rows(); i++)
			centersArr[i] = centers.get(i, 0)[0];
		Arrays.sort(centersArr);
		Imgproc.threshold(sub, sub, centersArr[centers.rows() - 1] / 2 + centersArr[centers.rows() - 2] / 2, 255,
				Imgproc.THRESH_BINARY);
		// Imgproc.threshold(sub, sub, 100, 255, Imgproc.THRESH_BINARY);
		// ImUtils.imshow(sub);
		Imgproc.erode(sub, sub, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5)));
		Imgproc.dilate(sub, sub, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5)));

		// ImUtils.imshow(sub);
		Imgproc.Canny(sub, sub, 50, 150);
		Imgproc.HoughLinesP(sub, outLines, 1, Math.PI / 180, 30, 50, 1000);

	}

	static void showSobel(Mat pic) {

		Mat tx = new Mat();
		Mat ty = new Mat();
		Imgproc.Sobel(pic, tx, CvType.CV_32F, 1, 0);
		Imgproc.Sobel(pic, ty, CvType.CV_32F, 0, 1);
		Core.absdiff(tx, new Scalar(0), tx);
		Core.absdiff(ty, new Scalar(0), ty);
		Core.add(tx, ty, tx);
		Core.normalize(tx, tx, 0, 255, Core.NORM_MINMAX);

		for (int r = 0; r < tx.rows(); r++)
			for (int c = 0; c < tx.cols(); c++) {
				double val = tx.get(r, c)[0];
				double s = 80;
				double d = Math.sqrt((r - pic.height() / 2) * (r - pic.height() / 2)
						+ (c - pic.width() / 2) * (c - pic.width() / 2));

				val *= Math.exp(-(d * d) / (s * s));
				// tx.put(r, c, val);
			}
		ImUtils.imshow(tx);
	}

	public static void main(String[] args) throws IOException {

		for (int N = 0; N < 1; N++) {
			Mat cam0_0 = Imgcodecs.imread(".\\" + N + "-0.jpg", Imgcodecs.IMREAD_GRAYSCALE);
			Mat cam0_1 = Imgcodecs.imread(".\\" + N + "-1.jpg", Imgcodecs.IMREAD_GRAYSCALE);
			Mat cam1_0 = Imgcodecs.imread(".\\" + (N + 1) + "-0.jpg", Imgcodecs.IMREAD_GRAYSCALE);
			Mat cam1_1 = Imgcodecs.imread(".\\" + (N + 1) + "-1.jpg", Imgcodecs.IMREAD_GRAYSCALE);
			Mat template = Imgcodecs.imread(".\\target-aruco.jpg", Imgcodecs.IMREAD_GRAYSCALE);

			// pre process and find lines
			Mat nodes = new Mat();
			Mat lines1 = new Mat(), lines2 = new Mat();
			Mat homo1 = new Mat(), homo2 = new Mat();
			findLines(cam0_0, cam1_0, template, lines1, homo1);
			findLines(cam0_1, cam1_1, template, lines2, homo2);
			Imgproc.cvtColor(template, template, Imgproc.COLOR_GRAY2BGR);
			// draw lines
			for (int i1 = 0; i1 < lines1.rows(); i1++)
				for (int i2 = 0; i2 < lines2.rows(); i2++) {
					double[] abc1 = getLineABC(lines1.row(i1));
					double[] abc2 = getLineABC(lines2.row(i2));
					Point node = new Point(
							(abc1[1] * abc2[2] - abc2[1] * abc1[2]) / (abc1[0] * abc2[1] - abc2[0] * abc1[1]),
							(abc1[0] * abc2[2] - abc2[0] * abc1[2]) / (abc2[0] * abc1[1] - abc1[0] * abc2[1]));
					Imgproc.line(template, new Point(0, -abc1[2] / abc1[1]),
							new Point(template.width(), (-template.width() * abc1[0] - abc1[2]) / abc1[1]),
							new Scalar(0, 255, 0));
					Imgproc.line(template, new Point(0, -abc2[2] / abc2[1]),
							new Point(template.width(), (-template.width() * abc2[0] - abc2[2]) / abc2[1]),
							new Scalar(0, 255, 0));
					if (!Double.isFinite(node.x) || !Double.isFinite(node.y) || node.x > template.width() || node.x < 0
							|| node.y > template.height() || node.y < 0)
						continue;

					Mat nodeMat = new Mat(1, 2, CvType.CV_32F);
					nodeMat.put(0, 0, node.x, node.y);
					nodes.push_back(nodeMat);
				}
			// ImUtils.imshow(template);
			// remove anomaly
			nodes = removeAnomalyNodes(nodes, 25);
			// kmeans to find anchor points
			TermCriteria tc = new TermCriteria(TermCriteria.MAX_ITER | TermCriteria.EPS, 100, 0.01);
			Mat centers = new Mat();
			Core.kmeans(nodes, 4, new Mat(), tc, 10, Core.KMEANS_PP_CENTERS, centers);
			Point center = new Point();
			for (int i = 0; i < 4; i++) {
				Imgproc.circle(template, new Point(centers.get(i, 0)[0], centers.get(i, 1)[0]), 2,
						new Scalar(255, 0, 0), 2);
				center.x += centers.get(i, 0)[0];
				center.y += centers.get(i, 1)[0];
			}
			center.x /= 4;
			center.y /= 4;
			Imgproc.circle(template, center, 1, new Scalar(0, 0, 255), 2);
			ImUtils.imshow(template);
			Mat angleVec = getInAngle(center, homo1, homo2, lines1, lines2);

			visualizeAngle(angleVec);

		}
	}

	static double[] getLineABC(Mat line) {
		Point p1 = new Point(line.get(0, 0)[0], line.get(0, 0)[1]);
		Point p2 = new Point(line.get(0, 0)[2], line.get(0, 0)[3]);
		double a = p2.y - p1.y, b = p1.x - p2.x, c = p2.x * p1.y - p1.x * p2.y;
		double factor = Math.sqrt(a * a + b * b);
		return new double[] { a / factor, b / factor, c / factor };
	}

	static double L2Distance(Point p1, Point p2) {
		return Math.sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
	}

	public static List<MyArUco> getMarkers(Mat pic) {
		List<MatOfPoint2f> markerPts = ard.findMarkers(pic);
		List<MyArUco> markers = new ArrayList<MyArUco>();
		for (int i = 0; i < markerPts.size(); i++) {
			MyArUco mm = MyArUco.parse(pic, markerPts.get(i));
			if (mm != null)
				markers.add(mm);
		}

		markers.sort(new Comparator<MyArUco>() {
			@Override
			public int compare(MyArUco arg0, MyArUco arg1) {
				return arg0.code2 - arg1.code2;
			}
		});
		return markers;
	}

	public static Mat getHomo(Mat pic, Mat template) {
		List<Point> srcPtsList = new ArrayList<Point>();
		List<Point> dstPtsList = new ArrayList<Point>();
		List<MyArUco> srcMarkers = getMarkers(template);
		List<MyArUco> dstMarkers = getMarkers(pic);
		for (int i = 0; i < 4; i++) {
			// System.out.println(srcMarkers.get(i).code2);
			srcPtsList.addAll(srcMarkers.get(i).pts.toList());
			dstPtsList.addAll(dstMarkers.get(i).pts.toList());
		}
		MatOfPoint2f srcPts = new MatOfPoint2f();
		MatOfPoint2f dstPts = new MatOfPoint2f();
		srcPts.fromList(srcPtsList);
		dstPts.fromList(dstPtsList);
		Mat homo = Calib3d.findHomography(srcPts, dstPts);

		return homo;
	}

	static Mat removeAnomalyNodes(Mat nodes, double threshold) {
		// every time removes the farthest point from center
		List<Point> nodeList = new LinkedList<Point>();
		Point center = new Point(0, 0);
		for (int i = 0; i < nodes.rows(); i++) {
			nodeList.add(new Point(nodes.get(i, 0)[0], nodes.get(i, 1)[0]));
			center.x += nodes.get(i, 0)[0];
			center.y += nodes.get(i, 1)[0];
		}
		center.x /= nodes.rows();
		center.y /= nodes.rows();
		double maxDistance = Double.MAX_VALUE;
		do {
			nodeList.sort(new Comparator<Point>() {
				@Override
				public int compare(Point p1, Point p2) {
					double d1 = L2Distance(p1, center);
					double d2 = L2Distance(p2, center);
					return -Double.compare(d1, d2);
				}
			});
			maxDistance = L2Distance(nodeList.get(0), center);
			if (maxDistance > threshold) {
				center.x *= nodeList.size();
				center.y *= nodeList.size();
				Point outLier = nodeList.remove(0);
				center.x = (center.x - outLier.x) / nodeList.size();
				center.y = (center.y - outLier.y) / nodeList.size();
			}
		} while (maxDistance > threshold);

		Mat result = new Mat(nodeList.size(), 2, CvType.CV_32F);
		for (int i = 0; i < nodeList.size(); i++) {
			result.put(i, 0, nodeList.get(i).x, nodeList.get(i).y);
		}
		return result;
	}

	static void visualizeAngle(Mat angleVec) {
		Mat visualizeAngleXY = Mat.ones(new Size(301, 301), CvType.CV_8U);
		Mat visualizeAngleZ = Mat.ones(new Size(301, 301), CvType.CV_8U);
		Core.multiply(visualizeAngleXY, new Scalar(255), visualizeAngleXY);
		Core.multiply(visualizeAngleZ, new Scalar(255), visualizeAngleZ);

		double cosz = angleVec.get(0, 0)[2];
		double sinz = Math.sqrt(1 - angleVec.get(0, 0)[2] * angleVec.get(0, 0)[2]);
		double cosx = angleVec.get(0, 0)[0] / sinz;
		double cosy = angleVec.get(0, 0)[1] / sinz;

		Imgproc.line(visualizeAngleXY, new Point(0, 150), new Point(300, 150), new Scalar(0));
		Imgproc.line(visualizeAngleXY, new Point(150, 0), new Point(150, 300), new Scalar(0));

		Imgproc.line(visualizeAngleXY, new Point(150, 150), new Point(150 + cosx * 150, 150 + cosy * 150),
				new Scalar(0));

		Imgproc.line(visualizeAngleZ, new Point(150, 0), new Point(150, 300), new Scalar(0));
		Imgproc.line(visualizeAngleZ, new Point(150, 300),
				new Point(150 + Math.signum(cosx) * 200 * sinz, 300 - 200 * cosz), new Scalar(0));

		Imgproc.putText(visualizeAngleXY,
				"thetaXY=" + new DecimalFormat("#.00").format(-Math.signum(cosy) * Math.acos(cosx) / Math.PI * 180),
				new Point(0, 10), Core.FONT_HERSHEY_PLAIN, 1, new Scalar(0));

		Imgproc.putText(visualizeAngleZ,
				"thetaZ=" + new DecimalFormat("#.00").format(Math.acos(angleVec.get(0, 0)[2]) / Math.PI * 180),
				new Point(0, 10), Core.FONT_HERSHEY_PLAIN, 1, new Scalar(0));
		ImUtils.imshow(visualizeAngleXY);
		ImUtils.imshow(visualizeAngleZ);
	}
}
