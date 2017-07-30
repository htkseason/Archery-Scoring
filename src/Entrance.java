import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;

import javax.swing.JFrame;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import pers.season.vml.ar.CameraData;
import pers.season.vml.ar.Engine3D;
import pers.season.vml.ar.FeatureTracker;
import pers.season.vml.ar.MotionFilter;
import pers.season.vml.util.*;

public final class Entrance {

	static {
		System.loadLibrary("lib/opencv_java320_x64");
	}

	static Mat findLines(Mat camx_0, Mat camx_1, Mat template) {
		FeatureTracker ft = new FeatureTracker();
		ft.setTemplate(template);
		Mat homo = ft.findHomo(camx_0, true, 0.8);
		Mat bg = new Mat();
		Mat bow = new Mat();
		Imgproc.warpPerspective(camx_0, bg, homo, template.size(), Imgproc.WARP_INVERSE_MAP);
		Imgproc.warpPerspective(camx_1, bow, homo, template.size(), Imgproc.WARP_INVERSE_MAP);

		Mat sub = new Mat();
		Core.absdiff(bg, bow, sub);
		Imgproc.medianBlur(sub, sub, 5);
		Imgproc.threshold(sub, sub, 0, 255, Imgproc.THRESH_OTSU);
		Imgproc.Canny(sub, sub, 50, 150);
		Mat lines = new Mat();
		Imgproc.HoughLinesP(sub, lines, 1, Math.PI / 180, 50, 100, 100);
		return lines;
	}

	static Mat removeAnomaly(Mat nodes, double threshold) {
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
				System.out.println(outLier);
			}
		} while (maxDistance > threshold);

		Mat result = new Mat(nodeList.size(), 2, CvType.CV_32F);
		for (int i = 0; i < nodeList.size(); i++) {
			result.put(i, 0, nodeList.get(i).x, nodeList.get(i).y);
		}
		return result;

	}

	public static void main(String[] args) throws IOException {
		Mat cam0_0 = Imgcodecs.imread("./cam0_0.jpg", Imgcodecs.IMREAD_GRAYSCALE);
		Mat cam0_1 = Imgcodecs.imread("./cam0_1.jpg", Imgcodecs.IMREAD_GRAYSCALE);
		Mat cam1_0 = Imgcodecs.imread("./cam1_0.jpg", Imgcodecs.IMREAD_GRAYSCALE);
		Mat cam1_1 = Imgcodecs.imread("./cam1_1.jpg", Imgcodecs.IMREAD_GRAYSCALE);

		Mat template = Imgcodecs.imread("./target.jpg");

		// pre process and find lines
		Mat nodes = new Mat();
		Mat lines1 = findLines(cam0_0, cam0_1, template);
		Mat lines2 = findLines(cam1_0, cam1_1, template);

		// draw lines
		for (int i1 = 0; i1 < lines1.rows(); i1++)
			for (int i2 = 0; i2 < lines2.rows(); i2++) {
				double[] kb1 = getLinekb(lines1.row(i1));
				double[] kb2 = getLinekb(lines2.row(i2));
				double k1 = kb1[0], b1 = kb1[1];
				double k2 = kb2[0], b2 = kb2[1];
				Point node = new Point((b2 - b1) / (k1 - k2), ((k1 * b2) - (k2 * b1)) / (k1 - k2));

				Imgproc.line(template, new Point(0, b1), new Point(template.width(), template.width() * k1 + b1),
						new Scalar(0, 255, 0));
				Imgproc.line(template, new Point(0, b2), new Point(template.width(), template.width() * k2 + b2),
						new Scalar(0, 255, 0));
				Mat nodeMat = new Mat(1, 2, CvType.CV_32F);
				nodeMat.put(0, 0, node.x, node.y);
				nodes.push_back(nodeMat);
			}

		// remove anomaly
		nodes = removeAnomaly(nodes, 25);

		// kmeans to find anchor points
		TermCriteria tc = new TermCriteria(TermCriteria.MAX_ITER | TermCriteria.EPS, 100, 0.01);
		Mat centers = new Mat();
		Core.kmeans(nodes, 4, new Mat(), tc, 10, Core.KMEANS_PP_CENTERS, centers);
		Point center = new Point();
		for (int i = 0; i < 4; i++) {
			Imgproc.circle(template, new Point(centers.get(i, 0)[0], centers.get(i, 1)[0]), 2, new Scalar(255, 0, 0),
					2);
			center.x += centers.get(i, 0)[0];
			center.y += centers.get(i, 1)[0];
		}
		center.x /= 4;
		center.y /= 4;
		Imgproc.circle(template, center, 2, new Scalar(0, 0, 255), 2);
		ImUtils.imshow(template);
	}

	static double[] getLinekb(Mat line) {
		Point p1 = new Point(line.get(0, 0)[0], line.get(0, 0)[1]);
		Point p2 = new Point(line.get(0, 0)[2], line.get(0, 0)[3]);
		double k = (p1.y - p2.y) / (p1.x - p2.x);
		double b = ((p1.y - k * p1.x) + (p2.y - k * p2.x)) / 2.;
		return new double[] { k, b };
	}

	static double L2Distance(Point p1, Point p2) {
		return Math.sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
	}

}
