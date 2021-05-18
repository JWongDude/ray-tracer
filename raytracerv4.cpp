#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>
#include <vector>
#include <string>
#include <unordered_map>
#include <random>
#include <execution>

// Macros and Random number generation
#define PI 3.1415926536

const int width=500, height=500;
const double inf=1e9;
const double eps=1e-6;

std::mt19937 mersenneTwister;
std::uniform_real_distribution<double> uniform;
#define RND (2.0*uniform(mersenneTwister)-1.0)
#define RND2 (uniform(mersenneTwister))

using namespace std;
typedef unordered_map<string, double> pl;

/* -------------Fundamental Vector Class----------------*/
struct Vec {
	double x, y, z;
	Vec(double x0=0, double y0=0, double z0=0){ x=x0; y=y0; z=z0; }

    //Addition/Subtraction
	Vec operator+(const Vec &b) const { return Vec(x+b.x,y+b.y,z+b.z); }
    Vec& operator+=(const Vec &b) { return *this = *this + b; }
	Vec operator-(const Vec &b) const { return Vec(x-b.x,y-b.y,z-b.z); }

    //Scalar Multipication/Division
	Vec operator*(double b) const { return Vec(x*b,y*b,z*b); }
	Vec operator/(double b) const { return Vec(x/b,y/b,z/b); }

    //Component-wise Multipication
	Vec mult(const Vec &b) const { return Vec(x*b.x,y*b.y,z*b.z); }

    //Length
	Vec& norm(){ return *this = *this * (1/sqrt(x*x+y*y+z*z)); }
	double length() { return sqrt(x*x+y*y+z*z); }

    //Products
	double dot(const Vec &b) const { return x*b.x+y*b.y+z*b.z; }
	Vec operator%(const Vec &b) const {return Vec(y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x);}
};

//Free vector functions for enhanced call syntax:
Vec mult(const Vec &v1, const Vec &v2) { 
    return Vec(v1.x*v2.x, v1.y*v2.y, v1.z*v2.z); 
}
double dot(const Vec& v1, const Vec& v2) {
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}
Vec& normalize(Vec&& v) {
    return v.norm();
}
double length(const Vec& v) {
    return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

/*-------------------Geometry----------------------*/
/* Ray representation.
Contains:
    -o: origin/location of ray
    -d: normalized direction of ray */
struct Ray {
	Vec o, d;
	Ray(Vec o0 = 0, Vec d0 = 0) { o = o0, d = d0.norm(); }
};

/* Object describes abstract hitible object. 
Contains:
    -Material Properties:
      -cl: defines albedo (reflectvity/absorbsion) of different wavelengths.
      -emission: defines emitted light of object. 
      -type: defines BRDF type: specular, diffuse, or refractive.

    -Intersection Routine
    -Normal Vector Calculation */
class Obj {
public:
	Vec cl;
	double emission;
	int type;
    void setMat(Vec cl_ = 0, double emission_ = 0, int type_ = 0) { cl=cl_; emission=emission_; type=type_; }
	
    //Object Specific:
    virtual double intersect(const Ray&) const = 0;
	virtual Vec normal(const Vec&) const = 0;
};

/* Plane Object.
Contains:
    n: normal vector 
    d: distance of plane from camera.
*/
class Plane : public Obj {
public:
	Vec n;
	double d;
	Plane(double d_ = 0, Vec n_ = {0, 0, 0})
        : n(n_), d(d_) {}
    
    /*
    Ray intersects plane if ray direction is not perfectly perpendicular 
    to the normal. Therefore, there is a vector on the plane from the hitpt 
    to the base of the normal: dot(r - r0, n) = 0.
    Substitute r for parametric form of ray: r = At + B, and solve for t.
    */
	double intersect(const Ray& ray) const {
		double d0 = dot(n, ray.d);
		if(d0 != 0) {
			double t = -1 * ((dot(ray.o, n) + d) / d0);
			return (t > eps) ? t : 0;
		}
		else return 0;
	}

    //Easy! Plane has uniform normal.
	Vec normal(const Vec& p0) const { return n; }
};

/* Sphere Object. 
Contains:
    -c: vector center
    -r: radius value */
class Sphere : public Obj {
public:
	Vec c;
	double r;

    /* Intersection with Sphere = Solving Quadratic Formula
    Discriminant Check (Vectors are d, o, c):
    A = d * d = 1
    B = 2d * (o - c)
    C = (o - c) * (o - c) - r**2 
    
    discrimValue = B**2 - 4C
    discrim = sqrt(discrimValue)

    After check, output intersect. Do not count self-intersect, sol2 = eps.*/
	Sphere(double r_= 0, Vec c_=0) { c=c_; r=r_; }
	double intersect(const Ray& ray) const {
		double b = dot(ray.d, (ray.o - c)*2);
		double c_ = dot((ray.o - c), (ray.o - c)) - (r*r);
		double disc = b*b - 4*c_;
		if (disc<0) return 0;
		else disc = sqrt(disc);
		double sol1 = -b + disc;
		double sol2 = -b - disc;
		return (sol2>eps) ? sol2/2 : ((sol1>eps) ? sol1/2 : 0);
	}

    //Subtract hitpoint and center vectors.
	Vec normal(const Vec& p0) const {
		return normalize(p0 - c);
	}
};

/*----------------Ray Tracing-------------------*/
/* Intersection Object.
Pairs time of intersection with the hit object. */
class Intersection {
public:
	double t;
	Obj* object;  //Abstract object cannot be instantiated, needs a pointer.
	Intersection() { t = inf; object = nullptr; }
	Intersection(double t_, Obj* object_) { t = t_; object = object_; }
	
    operator bool() { return object != nullptr; }
};

/* Scene Object.
Manages data structure of objects.
Calculates nearest intersection with object. */
class Scene {
private:
	vector<Obj*> objects;
public:
	void add(Obj* object) {
		objects.push_back(object);
	}

	Intersection intersect(const Ray& ray) const {
		Intersection closestIntersection;
		for (auto iter = objects.begin(); iter != objects.end(); ++iter) {
			double t = (*iter)->intersect(ray);
			if (t > eps && t < closestIntersection.t) {
				closestIntersection.t = t;
				closestIntersection.object = *iter;
			}
		}
		return closestIntersection;
	}
};

//Camera transform. 
//Maps 2D coordinate in 3D world space!
Vec camcr(const double x, const double y) {
	double w=width;
	double h=height;
	float fovx = PI/4;          //Horizontal 45 degrees 
	float fovy = (h/w) * fovx;  //Corresponding Vertical FOV wrt frame.
	return Vec(((2*x-w)/w) * tan(fovx), 
				-((2*y-h)/h) * tan(fovy),
				-1.0);          //Negligible
}

//Construct Orthonormal basis wrt to normalized v1.
//Used to calculate local light transport in diffuse BRDF.
void ons(const Vec& v1, Vec& v2, Vec& v3) {
    if (std::abs(v1.x) > std::abs(v1.y)) {
		// project to the y = 0 plane and construct a normalized orthogonal vector in this plane
		float invLen = 1.f / sqrtf(v1.x * v1.x + v1.z * v1.z);
		v2 = Vec(-v1.z * invLen, 0.0f, v1.x * invLen);
    } else {
		// project to the x = 0 plane and construct a normalized orthogonal vector in this plane
		float invLen = 1.0f / sqrtf(v1.y * v1.y + v1.z * v1.z);
		v2 = Vec(0.0f, v1.z * invLen, -v1.y * invLen);
    }
    v3 = v1 % v2;
}

// Uniform sampling on unit, positive hemisphere.
Vec hemisphere(double z, double u2) {
	const double r = sqrt(1.0-z*z);    //Legal length of 2D unit vector
	const double phi = 2 * PI * u2;    //Random angle offset
	return Vec(cos(phi)*r, sin(phi)*r, z);  
}

//Recursive ray tracer. Each recursive call stack represents rendering a single pixel.
void trace(Ray &ray, const Scene& scene, int depth, Vec& clr, pl& params) {
	
    //Russian roulette: starting at depth 5, 
    //each recursive step will stop with a probability of 0.1
	double rrFactor = 1.0;
	if (depth >= 5) {
		const double rrStopProbability = 0.1;
		if (RND2 <= rrStopProbability) {
			return;
		}
		rrFactor = 1.0 / (1.0 - rrStopProbability);
	}

    //Guard statement, quick return.
	Intersection intersection = scene.intersect(ray);
	if (!intersection) return;

	//Calculate hit point and update origin. 
	Vec hp = ray.o + ray.d * intersection.t;
	Vec N = intersection.object->normal(hp);
	ray.o = hp;

    //Retrieve direct illumination at intersection and 
    //BRDF for indirect illumination.
	const double emission = intersection.object->emission;
    Vec directIllum = Vec(emission, emission, emission) * rrFactor;
    int BRDF = intersection.object->type;
    switch (BRDF) {
        case 1: {
            //Define local coordinate basis wrt normal.
            Vec rotX, rotY;
            ons(N, rotX, rotY);

            //Generate random output direction on positive unit hemisphere.
            Vec sampledDir = hemisphere(RND2,RND2);
            
            //Map random direction to local coordinate space and update ray direction.
            //This is a matrix multiply!
            Vec rotatedDir;
            rotatedDir.x = dot(Vec(rotX.x, rotY.x, N.x), sampledDir);
            rotatedDir.y = dot(Vec(rotX.y, rotY.y, N.y), sampledDir);
            rotatedDir.z = dot(Vec(rotX.z, rotY.z, N.z), sampledDir);
            ray.d = rotatedDir;	//Normalized!

            //Store to calculate light attenutation
            //after we calculate indirect illumination.
            Vec current_direction = ray.d; 

            //Collect Indirect Illumination of future bounces.
            Vec indirectIllum;
            trace(ray,scene,depth+1,indirectIllum,params);
            
            //Calculate Rendering Equation.
            double cost= dot(N, current_direction);
            clr += directIllum + (indirectIllum.mult(intersection.object->cl)) * 0.1 * cost * rrFactor;
	    } break;   

	// Specular BRDF:
        case 2: {
            //Update Ray Direction, perfect reflection.
            double cost = dot(ray.d, N);
            ray.d = normalize(ray.d - N * (cost * 2));
            
            //Collect Indirect Illumination of future bounces.
            Vec indirectIllum = Vec(0,0,0);
            trace(ray,scene,depth+1,indirectIllum,params);
            
            //Calculate Rendering Equation.
            clr += directIllum + indirectIllum * rrFactor;
	    } break;

        //Glossy BRDF:
	    case 3: {
            //Snell's Law for refraction trajectory: 
            double n = params["refr_index"];
            double R0 = (1.0-n)/(1.0+n);
            R0 = R0*R0;
            if(dot(N, ray.d) >0) { // Inside medium
                N = N * -1;
                n = 1 / n;
            }
            n = 1 / n;

            //Fresnel's Law for probability of refraction.
            //Probability defines Bernoulli Distribution.:
            double cost1 = (dot(N, ray.d))*-1;          // cosine of theta_1
            double cost2 = 1.0 - n*n*(1.0-cost1*cost1); // cosine of theta_2
            double Rprob = R0 + (1.0-R0) * pow(1.0 - cost1, 5.0); // Schlick-approximation
            
            //Random Variable categorizes which trajectory.
            if (cost2 > 0 && RND2 > Rprob) { // refraction direction
                ray.d = normalize((ray.d*n)+(N*(n*cost1-sqrt(cost2))));
            }
            else { // reflection direction, same as specular BRDF trajectory.
                ray.d = normalize(ray.d+N*(cost1*2));
            }

            //Collect Indirect Illumination of future bounces.
            Vec indirectIllum;
            trace(ray,scene, depth+1, indirectIllum,params);

            //Calculate Rendering Equation.
            clr += directIllum + indirectIllum * 1.15 * rrFactor;
	    } break;
    }
}

int main() {
    //Set the Scene. Cornel Box, positive x/y/z are right, up, and out of the screen. 
	Scene scene;
	auto add=[&scene](Obj* s, Vec cl, double emission, int type) {
			s->setMat(cl,emission,type);
			scene.add(s);
	};

	// Radius, position, color, emission, type (1=diff, 2=spec, 3=refr) for spheres
	add(new Sphere(1.05,Vec(-0.75,-1.45,-4.4)),Vec(4,8,4),0,2); // Middle sphere
    //add(new Sphere(0.45,Vec(0.8,-2.05,-3.7)),Vec(10,10,1),0,3); // Right sphere
	add(new Sphere(0.5,Vec(2.0,-2.05,-3.7)),Vec(10,10,1),0,3); // Right sphere
	add(new Sphere(0.6,Vec(-1.75,-1.95,-3.1)),Vec(4,4,12),0,1); // Left sphere
	
    // Position, normal, color, emission, type for planes
	add(new Plane(2.5,Vec(0,1,0)),Vec(6,6,6),0,1); // Bottom plane
	add(new Plane(5.5,Vec(0,0,1)),Vec(6,6,6),0,1); // Back plane
	add(new Plane(2.75,Vec(1,0,0)),Vec(10,2,2),0,1); // Left plane
	add(new Plane(2.75,Vec(-1,0,0)),Vec(2,10,2),0,1); // Right plane
	add(new Plane(3.0,Vec(0,-1,0)),Vec(6,6,6),0,1); // Ceiling plane
	add(new Plane(0.5,Vec(0,0,-1)),Vec(6,6,6),0,1); // Front plane
	add(new Sphere(0.5,Vec(0,1.9,-3)),Vec(0,0,0),10000,1); // Light

    //Initalize Params:
	pl params;
	params["refr_index"] = 1.5;     // For Glossy BRDF, glass constant.
	params["spp"] = 100.0;           // samples per pixel
	double spp = params["spp"];
	
    //Initalize Picture Frame.
    //First, store row of Vector pointers.
    //Then, fill in Vector pointers with Vector arrays. 
	Vec **pix = new Vec*[width];
	for(int i=0;i<width;i++) {
		pix[i] = new Vec[height];
	}

    //Start Stop Watch, Begin Render.
	srand(time(NULL));
	clock_t start = clock();

    #pragma omp parallel for collapse(3)
	for (int col = 0; col < width; col++) {
        for(int row = 0; row < height; row++) {		
            //Shoot spp rays per pixel. Intuitively, need to cast multiple rays
            //to approximate light hitting the hemisphere in all directions.
			for(int s = 0; s < spp; s++) {
                //Init new color and ray.
				Vec color;
				Ray ray;
				ray.o = (Vec(0,0,0));

                //Calculate inital trajectory of ray.
				Vec cam = camcr(col,row);  //Find 3D world space coordinates.
				cam.x = cam.x + RND/800;   //Random ray offsets for anti-aliasing.
				cam.y = cam.y + RND/800;
				ray.d = normalize(cam - ray.o);  //Set the ray direction.
				
                //Calculate the color.
                trace(ray,scene,0,color,params);
				
                //Average color contributions into image pixel.
                pix[col][row] = pix[col][row] + color / spp;
			}
		}
	}

    //Print PPM file!
	FILE *f = fopen("ray5.ppm", "w");
	fprintf(f, "P3\n%d %d\n%d\n ", width, height, 255);
	for (int row = 0; row < height; row++) {
		for (int col=0;col<width;col++) {
			fprintf(f,"%d %d %d ", min((int)pix[col][row].x,255), 
                                min((int)pix[col][row].y,255), 
                                min((int)pix[col][row].z,255));
		}
		fprintf(f, "\n");
	}
	fclose(f);

    //End Stop Watch:
	clock_t end = clock();
	double t = (double)(end-start)/CLOCKS_PER_SEC;
	printf("\nRender time: %fs.\n",t);
	return 0;
}

//Parallelize. Understand plane and make scene. 
//Do git tomorrow?