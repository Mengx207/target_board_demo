#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include <iostream>
using namespace cv;
using namespace std;
vector<Point3f> createBoardPoints(Size2i board_shape, double diagonal_spacing);

int main(int argc, char ** argv)
{
    Mat gray = imread( "image_captured.png", IMREAD_GRAYSCALE);
    // Mat gray = imread( "symmetric_dots.png", IMREAD_GRAYSCALE);
    Mat draw_img(gray.rows, gray.cols, IMREAD_GRAYSCALE);

    if( gray.empty()){
        cout << "Error opening image" << endl;
        return EXIT_FAILURE;
    }
    Size patternsize(5,11); // how to define the size of asymmetric pattern?
    vector<Point2f> centers; // center of feature dots
    SimpleBlobDetector::Params params;
    params.maxArea = 10e4;
    // Ptr<FeatureDetector> blobDetector = new SimpleBlobDetector(params);
    Ptr<FeatureDetector> blobDetector = SimpleBlobDetector::create(params);
    bool patternfound = findCirclesGrid(gray,patternsize,centers, CALIB_CB_ASYMMETRIC_GRID, blobDetector);
    // cout<<centers<<endl;
    // cout<<centers.size()<<endl;
    drawChessboardCorners(draw_img, patternsize, Mat(centers), patternfound);

    Size2i board_shape(5,11);
    double diagonal_spacing = 9;
    vector<Point3f>boardPoints = createBoardPoints(board_shape, diagonal_spacing);
    cout<<boardPoints<<endl<<boardPoints.size()<<endl;

    imshow("Captured Image", gray);    // Show the result
    imshow("Found Centers", draw_img);
    waitKey();
    return EXIT_SUCCESS;
}

vector<Point3f> createBoardPoints(Size2i board_shape, double diagonal_spacing)
{
    double spacing = diagonal_spacing/sqrt(2);
    vector<Point3f> centered_board_points(board_shape.area());
    for(int n=0; n<centered_board_points.size(); ++n)
    {
        int row_n = n/board_shape.width;
        int col_n = n%board_shape.width;
        centered_board_points[n].x = (float)(2*col_n+row_n%2+0.5-board_shape.width)*spacing;
        centered_board_points[n].y = (float)(row_n - (board_shape.height - 1) / 2.) * spacing;
        centered_board_points[n].z = 0.0;
    } 
    return centered_board_points;

}
