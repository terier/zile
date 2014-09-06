/*
 *  OpenCL Raycaster - volumetric data visualization application
 *  Copyright (C) 2014  Ziga Lesar
 *  
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *  
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *  OpenCL raycaster, written for my bachelor's thesis.
 *  I'm not proud of the structure of the program nor of the overall readability,
 *  but it gets the job done. This is my first time programming in OpenCL
 *  and the first time writing anything for parallel execution. I know that most
 *  of the code could be optimized further, but as I'm running out of time for
 *  implementation this is the best I could come up with.
 *
 *  Ziga Lesar, september 2014
 */

#ifdef USE_TEXTURE
    #define OUTPUT_TYPE write_only image2d_t
	#define INPUT_TYPE read_only image2d_t
#else
    #define OUTPUT_TYPE global uint *
	#define INPUT_TYPE global const uint *
#endif

#define INV255 0.00392156862f

#define MAX_ITER 500
#define SPARSE_SAMPLING_INTERPOLATION_THRESHOLD 0.5f // for sparse sampling

#define SIMPLE_SAMPLING // don't use interpolation on low values
#define THRESHOLD_MARGIN (threshold*0.9f)

#define ADAPTIVE_SAMPLING

#define AO_DEPTH 100.f
#define SSAO_SAMPLES 16

//#define MIP // undef for alpha


// nearest-neighbour sampler for texture-based read functions
constant const sampler_t nearest_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;


// kernels for gauss, thank you asodja

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Gaussian filter in X direction.                                           //
//                                                                           //
// params:                                                                   //
//   matrix       - data matrix to filter                                    //
//   Nx, Ny, Nz   - matrix dimensions                                        //
//   size         - convolution kernel size                                  //
//   gaussKernel  - convolution kernel                                       //
//   result       - result buffer                                            //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
kernel void gaussX(global const float* matrix, const int Nx, const int Ny, const int Nz,
const int size, global const float* gaussKernel, global float* result
) {
	// dimensions
   	const int x = get_global_id(0);
   	const int y = get_global_id(1);
   	const int z = get_global_id(2);
   	
   	float value = 0;
   	int xVar = 0;
	for (int i = 0; i < size; i++) {
		xVar = x + i - size / 2;
		if (xVar < 0 || xVar >= Nx)
			continue;
		value += matrix[xVar + y * Nx + z * Nx * Ny] * gaussKernel[size - i - 1];
	}
	result[x + y * Nx + z * Nx * Ny] = value;
}

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Gaussian filter in Y direction.                                           //
//                                                                           //
// params:                                                                   //
//   matrix       - data matrix to filter                                    //
//   Nx, Ny, Nz   - matrix dimensions                                        //
//   size         - convolution kernel size                                  //
//   gaussKernel  - convolution kernel                                       //
//   result       - result buffer                                            //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
kernel void gaussY(global const float* matrix, const int Nx, const int Ny, const int Nz,
const int size, global const float* gaussKernel, global float* result
) {
	// dimensions
   	const int x = get_global_id(0);
   	const int y = get_global_id(1);
   	const int z = get_global_id(2);
   	
   	float value = 0;
   	int yVar = 0;
	for (int i = 0; i < size; i++) {
		yVar = y + i - size / 2;
		if (yVar < 0 || yVar >= Ny)
			continue;
		value += matrix[x + yVar * Nx + z * Nx * Ny] * gaussKernel[size - i - 1];
	}
	result[x + y * Nx + z * Nx * Ny] = value;
}

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Gaussian filter in Z direction.                                           //
//                                                                           //
// params:                                                                   //
//   matrix       - data matrix to filter                                    //
//   Nx, Ny, Nz   - matrix dimensions                                        //
//   size         - convolution kernel size                                  //
//   gaussKernel  - convolution kernel                                       //
//   result       - result buffer                                            //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
kernel void gaussZ(global const float* matrix, const int Nx, const int Ny, const int Nz,
const int size, global const float* gaussKernel, global float* result
) {
	// dimensions
   	const int x = get_global_id(0);
   	const int y = get_global_id(1);
   	const int z = get_global_id(2);
   	
	float value = 0;
	int zVar = 0;
	for (int i = 0; i < size; i++) {
		zVar = z + i - size / 2;
		if (zVar < 0 || zVar >= Nz)
			continue;
		value += matrix[x  + y * Nx + zVar * Nx * Ny] * gaussKernel[size - i - 1];
	}
	result[x + y * Nx + z * Nx * Ny] = value;
}











///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Generic 2D convolution function.                                          //
//                                                                           //
// params:                                                                   //
//   input     - input image                                                 //
//   output    - output image                                                //
//   mask      - convolution kernel                                          //
//   maskSize  - convolution kernel size                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
kernel void imageConvolution(INPUT_TYPE input, OUTPUT_TYPE output, global const float* mask, const int maskSize) {
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int w = get_global_size(0);
	const int h = get_global_size(1);
	const int s = maskSize / 2;

	float4 color = (float4) 0;

	#ifdef USE_TEXTURE
		for (int i=0; i<maskSize; i++) {
			for (int j=0; j<maskSize; j++) {
				color += read_imagef(input, nearest_sampler, (int2)(x + i - s, y + j - s)) * mask[i + maskSize * j];
			}
		}
		write_imagef(output, (int2)(x, y), color);
	#else
		for (int i=0; i<maskSize; i++) {
			for (int j=0; j<maskSize; j++) {
				int value = input[x + i - s + w * (y + j - s)];
				float4 val = (float4)((float)(value & 0xFF) * INV255, (float)(value >> 8 & 0xFF) * INV255, (float)(value >> 16 & 0xFF) * INV255, 1.f);
				color += val * mask[i + maskSize * j];
			}
		}
		writeColorUint(output, x + w * y, color);
	#endif
}

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Generic 2D convolution function in X direction.                           //
//                                                                           //
// params:                                                                   //
//   input     - input image                                                 //
//   output    - output image                                                //
//   mask      - convolution kernel                                          //
//   maskSize  - convolution kernel size                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
kernel void imageConvolutionX(INPUT_TYPE input, OUTPUT_TYPE output, global const float* mask, const int maskSize) {
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int w = get_global_size(0);
	const int h = get_global_size(1);
	const int s = maskSize / 2;

	float4 color = (float4) 0;

	#ifdef USE_TEXTURE
		for (int i=0; i<maskSize; i++) {
			color += read_imagef(input, nearest_sampler, (int2)(x + i - s, y)) * mask[i];
		}
		write_imagef(output, (int2)(x, y), color);
	#else
		for (int i=0; i<maskSize; i++) {
			int xx = clamp(x + i - s, 0, w - 1);
			int yy = clamp(y, 0, h - 1);
			int value = input[xx + w * yy];
			float4 val = (float4)((float)(value & 0xFF) * INV255, (float)(value >> 8 & 0xFF) * INV255, (float)(value >> 16 & 0xFF) * INV255, 1.f);
			color += val * mask[i];
		}
		writeColorUint(output, x + w * y, color);
	#endif
}

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Generic 2D convolution function in Y direction.                           //
//                                                                           //
// params:                                                                   //
//   input     - input image                                                 //
//   output    - output image                                                //
//   mask      - convolution kernel                                          //
//   maskSize  - convolution kernel size                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
kernel void imageConvolutionY(INPUT_TYPE input, OUTPUT_TYPE output, global const float* mask, const int maskSize) {
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int w = get_global_size(0);
	const int h = get_global_size(1);
	const int s = maskSize / 2;

	float4 color = (float4) 0;

	#ifdef USE_TEXTURE
		for (int i=0; i<maskSize; i++) {
			color += read_imagef(input, nearest_sampler, (int2)(x, y + i - s)) * mask[i];
		}
		write_imagef(output, (int2)(x, y), color);
	#else
		for (int i=0; i<maskSize; i++) {
			int xx = clamp(x, 0, w - 1);
			int yy = clamp(y + i - s, 0, h - 1);
			int value = input[xx + w * yy];
			float4 val = (float4)((float)(value & 0xFF) * INV255, (float)(value >> 8 & 0xFF) * INV255, (float)(value >> 16 & 0xFF) * INV255, 1.f);
			color += val * mask[i];
		}
		writeColorUint(output, x + w * y, color);
	#endif
}


///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Generic image copy kernel. Comes in handy when working with a limited     //
// number of image buffers.                                                  //
//                                                                           //
// params:                                                                   //
//   input     - input image                                                 //
//   output    - output image                                                //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
kernel void imageCopy(INPUT_TYPE input, OUTPUT_TYPE output) {
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	#ifdef USE_TEXTURE
		write_imagef(output, (int2)(x, y), read_imagef(input, nearest_sampler, (int2)(x, y)));
	#else
		output[x + w * y] = input[x + w * y];
	#endif
}






///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Reads depth information from an image. Evil floating point bit-level      //
// hacking for the purpose of encoding floating point information in a       //
// component-normalized float4.                                              //
//                                                                           //
// params:                                                                   //
//   depthBuffer  - depth image                                              //
//   pos          - sampling position                                        //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
inline float readDepth(INPUT_TYPE depthBuffer, int2 pos) {
	#ifdef USE_TEXTURE
	int4 d = convert_int4(read_imagef(depthBuffer, nearest_sampler, pos) * 255.f);
	return as_float(d.x << 24 | d.y << 16 | d.z << 8 | d.w);
	#endif
}

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Writes depth information to an image. Evil floating point bit-level       //
// hacking for the purpose of encoding floating point information in a       //
// component-normalized float4.                                              //
//                                                                           //
// params:                                                                   //
//   depthBuffer  - depth image                                              //
//   pos          - sampling position                                        //
//   depth        - depth value to write                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
inline void writeDepth(OUTPUT_TYPE depthBuffer, int2 pos, float depth) {
	#ifdef USE_TEXTURE
	int dint = as_int(depth);
	write_imagef(depthBuffer, pos,
		convert_float4((int4)(dint >> 24 & 0xFF, dint >> 16 & 0xFF, dint >> 8 & 0xFF, dint & 0xFF)) * INV255);
	#endif
}


///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Reads color information from a PBO image and returns a                    //
// component-normalized float4.                                              //
//                                                                           //
// params:                                                                   //
//   input  - input image                                                    //
//   idx    - pixel index                                                    //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
inline float4 readColorUint(INPUT_TYPE input, int idx) {
	#ifdef USE_TEXTURE
	return (float4)0;
	#else
	uint c = input[idx];
	return (float4)((float)(c & 0xFF) * INV255, (float)(c >> 8 & 0xFF) * INV255, (float)(c >> 16 & 0xFF) * INV255, 1.f);
	#endif
}

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Writes color information to a PBO image.                                  //
//                                                                           //
// params:                                                                   //
//   output  - output image                                                  //
//   idx     - pixel index                                                   //
//   color   - color in component-normalized float4                          //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
inline void writeColorUint(OUTPUT_TYPE output, int idx, float4 color) {
	#ifndef USE_TEXTURE
	output[idx] = (int)(color.x * 255.f) | (int)(color.y * 255.f) << 8 | (int)(color.z * 255.f) << 16;
	#endif
}





///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Linear interpolation of two values.                                       //
//                                                                           //
// params:                                                                   //
//   a - first data value                                                    //
//   b - second data value                                                   //
//   t - interpolation factor                                                //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
inline float lerp(float a, float b, float t) {
	return a + t * (b - a);
}

inline float3 lerp3(float3 a, float3 b, float t) {
	return a + t * (b - a);
}

inline float4 lerp4(float4 a, float4 b, float t) {
	return a + t * (b - a);
}



///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Generate orthogonal subspace from input 3D vector.                        //
//                                                                           //
// params:                                                                   //
//   a  - input vector                                                       //
//   v1 - 1st output vector                                                  //
//   v2 - 2nd output vector                                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
inline void orthogonalSubspace(const float3 a, float3* v1, float3* v2) {
	float ax = fabs(a.x);
	float ay = fabs(a.y);
	float az = fabs(a.z);
	// quality subspace calculation: choose absolute biggest component
	if (ax > ay) {
		if (ax > az) {
			*v1 = (float3)(a.y, -a.x, 0);
		} else {
			*v1 = (float3)(-a.z, 0, a.x);
		}
	} else {
		if (ay > az) {
			*v1 = (float3)(0, a.z, -a.y);
		} else {
			*v1 = (float3)(-a.z, 0, a.x);
		}
	}
	*v2 = cross(a, *v1);
}




///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Samples the data matrix and returns sampled value.                        //
// Performs trilinear interpolation between adjacent values.                 //
//                                                                           //
// params:                                                                   //
//   data   - input volume data                                              //
//   M      - matrix array dimensions                                        //
//   invD   - inverse of voxel dimensions                                    //
//   point  - sampling point                                                 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
inline float sample(global const float* data, const int3 M, const float3 invD, float3 point) {
	// float & int indices
	float3 idfp = clamp(point * invD, (float3)0, convert_float3(M) - (float3)1.0001f);
	int3 id = convert_int3(idfp);
	// fractional parts for interpolation
	float3 fract = idfp - floor(idfp);
	
	int a = M.x; // index offset in y direction
	int b = M.x * M.y; // index offset in z direction
	
	int idx = id.x + a*id.y + b*id.z;
	float v000 = data[idx];
	float v001 = data[idx + b];
	float v010 = data[idx + a];
	float v011 = data[idx + a + b];
	float v100 = data[idx + 1];
	float v101 = data[idx + 1 + b];
	float v110 = data[idx + 1 + a];
	float v111 = data[idx + 1 + a + b];
	
	float v00 = lerp(v000, v001, fract.z);
	float v01 = lerp(v010, v011, fract.z);
	float v10 = lerp(v100, v101, fract.z);
	float v11 = lerp(v110, v111, fract.z);
	
	float v0 = lerp(v00, v01, fract.y);
	float v1 = lerp(v10, v11, fract.y);
	
	return lerp(v0, v1, fract.x);
}



///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Samples the data matrix and returns sampled value.                        //
// Performs plain sampling, without interpolation.                           //
//                                                                           //
// params:                                                                   //
//   data   - input volume data                                              //
//   M      - matrix array dimensions                                        //
//   invD   - inverse of voxel dimensions                                    //
//   point  - sampling point                                                 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
inline float sampleSimple(global const float* data, const int3 M, const float3 invD, float3 point) {
	int3 idx = clamp(convert_int3(point * invD), (int3)0, M - (int3)1);
	return data[idx.x + M.x*idx.y + M.x*M.y*idx.z];
}



///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Samples the data matrix and returns sampled normal.                       //
// Performs trilinear interpolation between adjacent normals.                //
//                                                                           //
// params:                                                                   //
//   data   - input volume data                                              //
//   M      - matrix array dimensions                                        //
//   invD   - inverse of voxel dimensions                                    //
//   point  - sampling point                                                 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
inline float3 sampleNormal(global const float* data, const int3 M, const float3 invD, float3 point) {
	// float indices
	float3 idfp = clamp(point * invD, (float3)0, convert_float3(M) - (float3)1.0001f);

	// indices
	int i1 = (int) idfp.x;
	int j1 = (int) idfp.y;
	int k1 = (int) idfp.z;
	int i0 = max(i1 - 1, 0);
	int j0 = max(j1 - 1, 0);
	int k0 = max(k1 - 1, 0);
	int i2 = min(i1 + 1, M.x - 1);
	int j2 = min(j1 + 1, M.y - 1);
	int k2 = min(k1 + 1, M.z - 1);
	int i3 = min(i1 + 2, M.x - 1);
	int j3 = min(j1 + 2, M.y - 1);
	int k3 = min(k1 + 2, M.z - 1);
	
	// fractional parts for interpolation
	float3 fract = idfp - floor(idfp);
	
	int a = M.x; // index offset in y direction
	int b = M.x * M.y; // index offset in z direction
	
	float v111 = data[i1 + a*j1 + b*k1];
	float v112 = data[i1 + a*j1 + b*k2];
	float v121 = data[i1 + a*j2 + b*k1];
	float v122 = data[i1 + a*j2 + b*k2];
	float v211 = data[i2 + a*j1 + b*k1];
	float v212 = data[i2 + a*j1 + b*k2];
	float v221 = data[i2 + a*j2 + b*k1];
	float v222 = data[i2 + a*j2 + b*k2];
	float v011 = data[i0 + a*j1 + b*k1];
	float v012 = data[i0 + a*j1 + b*k2];
	float v021 = data[i0 + a*j2 + b*k1];
	float v022 = data[i0 + a*j2 + b*k2];
	float v101 = data[i1 + a*j0 + b*k1];
	float v102 = data[i1 + a*j0 + b*k2];
	float v201 = data[i2 + a*j0 + b*k1];
	float v202 = data[i2 + a*j0 + b*k2];
	float v110 = data[i1 + a*j1 + b*k0];
	float v120 = data[i1 + a*j2 + b*k0];
	float v210 = data[i2 + a*j1 + b*k0];
	float v220 = data[i2 + a*j2 + b*k0];
	float v311 = data[i3 + a*j1 + b*k1];
	float v312 = data[i3 + a*j1 + b*k2];
	float v321 = data[i3 + a*j2 + b*k1];
	float v322 = data[i3 + a*j2 + b*k2];
	float v131 = data[i1 + a*j3 + b*k1];
	float v132 = data[i1 + a*j3 + b*k2];
	float v231 = data[i2 + a*j3 + b*k1];
	float v232 = data[i2 + a*j3 + b*k2];
	float v113 = data[i1 + a*j1 + b*k3];
	float v123 = data[i1 + a*j2 + b*k3];
	float v213 = data[i2 + a*j1 + b*k3];
	float v223 = data[i2 + a*j2 + b*k3];
	
	float3 n000 = (float3)(v211 - v011, v121 - v101, v112 - v110);
	float3 n001 = (float3)(v212 - v012, v122 - v102, v113 - v111);
	float3 n010 = (float3)(v221 - v021, v131 - v111, v122 - v120);
	float3 n011 = (float3)(v222 - v022, v132 - v112, v123 - v121);
	float3 n100 = (float3)(v311 - v111, v221 - v201, v212 - v210);
	float3 n101 = (float3)(v312 - v112, v222 - v202, v213 - v211);
	float3 n110 = (float3)(v321 - v121, v231 - v211, v222 - v220);
	float3 n111 = (float3)(v322 - v122, v232 - v212, v223 - v221);
	
	float3 n00 = lerp3(n000, n001, fract.z);
	float3 n01 = lerp3(n010, n011, fract.z);
	float3 n10 = lerp3(n100, n101, fract.z);
	float3 n11 = lerp3(n110, n111, fract.z);
	
	float3 n0 = lerp3(n00, n01, fract.y);
	float3 n1 = lerp3(n10, n11, fract.y);
	
	return lerp3(n0, n1, fract.x);
}




///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Samples the data matrix and returns sampled normal.                       //
// Performs trilinear interpolation between adjacent normals.                //
//                                                                           //
// params:                                                                   //
//   a          - low bound, smaller than threshold                          //
//   b          - high bound, larger than threshold                          //
//   data       - input volume data                                          //
//   M          - matrix array dimensions                                    //
//   invD   - inverse of voxel dimensions                                    //
//   iter       - number of iterations                                       //
//   threshold  - zero crossing                                              //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
float3 regulaFalsi(float3 a, float3 b, global const float* data,
const int3 M, const float3 invD,
const int iter, const float threshold
) {
	if (iter <= 0)
		return b;

	float3 c;
	float fa = sample(data, M, invD, a) - threshold;
	float fb = sample(data, M, invD, b) - threshold;
	float fc;
	for (int i=0; i<iter; i++) {
		c = b - (fb / (fb - fa)) * (b - a);
		//c = a + (b - a) * 0.5f;
		fc = sample(data, M, invD, c) - threshold;
		if (fc * fa < 0) {
			b = c;
			fb = fc;
		} else {
			a = c;
			fa = fc;
		}
	}
	
	return c;
}





///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Samples the data matrix to estimate the ambient occlusion integral.       //
//                                                                           //
// params:                                                                   //
//   point      - intersection point                                         //
//   v1         - normal vector                                              //
//   data       - input volume data                                          //
//   M          - matrix array dimensions                                    //
//   invD   - inverse of voxel dimensions                                    //
//   threshold  - isovalue                                                   //
//   length     - estimation hemisphere radius                               //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
float ambientOcclusion(const float3 point, float3 v1, global const float* data,
const int3 M, const float3 invD,
const float threshold, const float length
) {
	float3 v2, v3, v;
	orthogonalSubspace(v1, &v2, &v3);
	v1 = v1 * length;
	v2 = fast_normalize(v2) * length;
	v3 = fast_normalize(v3) * length;
	
	float n = 0.f;
	float s = 0.f;
	
	for (int i=-2; i<=2; i++) {
		for (int j=-2; j<=2; j++) {
			for (int k=1; k<=3; k++) {
				v = (float)i * v2 + (float)j * v3 + (float)k * v1;
				if (sample(data, M, invD, point + v) > threshold) // without interpolation - same image quality
					s += dot(fast_normalize(v1), fast_normalize(v));
				n += 1.f;
			}
		}
	}
	
	return s / n;
}





///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Computes ray-box intersections and returns true                           //
// if there are intersections.                                               //
// A point on ray at time lambda is calculated as S + lambda * E.            //
//                                                                           //
// params:                                                                   //
//   S        - ray start                                                    //
//   E        - ray direction                                                //
//   boxmin   - min box corner                                               //
//   boxmax   - max box corner                                               //
//   result   - pointer to buffer, where intersection times are stored       //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
inline bool raybox(float3 S, float3 E, float3 boxmin, float3 boxmax, float2* result) {
	float lambdamin, lambdamax;
	float lambda1, lambda2, lambda3, lambda4, lambda5, lambda6;
	float3 invE = 1 / E;
	
	if (E.x < 0) {
		lambda2 = (boxmin.x - S.x) * invE.x;
		lambda1 = (boxmax.x - S.x) * invE.x;
	} else {
		lambda1 = (boxmin.x - S.x) * invE.x;
		lambda2 = (boxmax.x - S.x) * invE.x;
	}
	
	if (E.y < 0) {
		lambda4 = (boxmin.y - S.y) * invE.y;
		lambda3 = (boxmax.y - S.y) * invE.y;
	} else {
		lambda3 = (boxmin.y - S.y) * invE.y;
		lambda4 = (boxmax.y - S.y) * invE.y;
	}
	
	lambdamin = fmax(lambda1, lambda3);
	lambdamax = fmin(lambda2, lambda4);
	if (lambdamin > lambdamax) return false;
	
	if (E.z < 0) {
		lambda6 = (boxmin.z - S.z) * invE.z;
		lambda5 = (boxmax.z - S.z) * invE.z;
	} else {
		lambda5 = (boxmin.z - S.z) * invE.z;
		lambda6 = (boxmax.z - S.z) * invE.z;
	}
	
	lambdamin = fmax(lambdamin, lambda5);
	lambdamax = fmin(lambdamax, lambda6);
	if (lambdamin > lambdamax) return false;
	(*result).x = lambdamin;
	(*result).y = lambdamax;
	return true;
}




///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Raycasting algorithm. Used in raycasting kernels without                  //
// alpha compositing.                                                        //
//                                                                           //
// params:                                                                   //
//   S          - start of ray                                               //
//   E          - direction of ray                                           //
//   matrix     - input volume data                                          //
//   M          - matrix array dimensions                                    //
//   D          - volume dimensions                                          //
//   threshold  - isosurface value                                           //
//   lin        - generic debugging int                                      //
//   depth      - pointer to depth buffer                                    //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
float4 cast(
float3 S, float3 E,
global const float* matrix,
const int3 M, const float3 D,
const float threshold, const int lin,
float* depth
) {
	*depth = FLT_MAX;
	float4 color = (float4)(0,0,0,0); // output color
	const float4 background = (float4)(0.3,0.3,0.3,1);
	const float4 debugcolor = (float4)(0.4,0.4,0.4,1);
	
	// shading params
	const float4 diffuse = (float4)(1,0,0,1);
	const float4 specular = (float4)(1,1,1,1);
	const float specularity = 50.f;
	const float aostrength = 2.f;
	
	// volume dimensions
	const float3 size = D * convert_float3(M);
	const float3 invD = 1 / D;
	
	
	// raycasting aux data
	float dataValue = 0;
	float3 P, Pprev, step;
	float cellSize = fmin(fmin(D.x, D.y), D.z);
	
	#ifdef SIMPLE_SAMPLING
	int simpleSampling = 1;
	int check = 0;
	#endif
	
	
	float2 result = (float2)(0,0); // raycasting lambda result buffer
	if (raybox(S, E, (float3)(0,0,0), size, &result) && result.y > 0) {
		result.x = fmax(result.x, 0); // start ray from camera, not from behind the camera
		
		
		
		
		float3 L = S;
		
		P = S + result.x * E; // current sampling position
		Pprev = P;
		
		int i = 0;
		int imax = (result.y - result.x) / cellSize; // max number of iterations
		step = ((result.y - result.x) / (float) imax) * E; // sampling interval
		while (i < imax && i < MAX_ITER) {
			#ifdef SIMPLE_SAMPLING
			if (simpleSampling) {
				dataValue = sampleSimple(matrix, M, invD, P);
				if (dataValue > threshold - THRESHOLD_MARGIN) {
					simpleSampling = 0;
					check = 0;
					continue;
				}
			} else {
				dataValue = sample(matrix, M, invD, P);
				if (check && dataValue < threshold - THRESHOLD_MARGIN) {
					simpleSampling = 1;
					check = 0;
					continue;
				}
			}
			check = 1;
			#else
			dataValue = sample(matrix, M, invD, P);
			#endif
			
			#ifdef SIMPLE_SAMPLING
			if (!simpleSampling && dataValue > threshold) {
			#else
			if (dataValue > threshold) {
			#endif
				// intersection point refinement
				P = regulaFalsi(Pprev, P, matrix, M, invD, 5, threshold);
				*depth = fast_distance(S, P);
			
				// gradient estimate
				float3 Nv = fast_normalize(sampleNormal(matrix, M, invD, P));
				
				// shade
				float3 Lv = fast_normalize(L - P); // light vector
				float3 Lrv = (2.f * dot(Lv, Nv)) * Nv - Lv; // reflected light vector
				float3 Cv = -E; // camera vector
				float dk = clamp(dot(-Lv, Nv), 0.f, 1.f); // diffuse coef
				float sk = clamp(pow(fmax(dot(Cv, Lrv), 0.f), specularity), 0.f, 1.f); // specular coef
				//if (lin == 1 && fast_length(P - S) < cellSize * AO_DEPTH) {
					float ao = ambientOcclusion(P, -Nv, matrix, M, invD, threshold, cellSize);
					color = (diffuse * dk + specular * sk) * fmax(1.f - aostrength * ao, 0.f);
				//} else {
					//color = diffuse * dk + specular * sk;
				//}
				break;
			}
		
			Pprev = P;
			P += step;
			i++;
		}
		
		if (i == imax) {
			color = background;
		}
	
	
	
	
	} else {
		if (result.y < 0)
			color = background;
		else
			color = debugcolor;
	}
	
	return color;
}






///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Raycasting algorithm. Used in raycasting kernels using                    //
// alpha compositing.                                                        //
//                                                                           //
// params:                                                                   //
//   S           - start of ray                                              //
//   E           - direction of ray                                          //
//   matrix      - input volume data                                         //
//   M           - matrix array dimensions                                   //
//   D           - volume dimensions                                         //
//   trf         - transfer function - mapping from reals to color/alpha     //
//   trfSamples  - transfer function array size                              //
//   threshold   - isosurface value                                          //
//   lin         - generic debugging int                                     //
//   depth       - pointer to depth buffer                                   //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
float4 castAlpha(
float3 S, float3 E,
global const float* matrix,
const int3 M, const float3 D,
global const float4* trf, const int trfSamples,
const float threshold, const int lin,
float* depth
) {	
	float4 color = (float4)(0,0,0,0); // output color
	float4 currentColor;
	const float4 background = (float4)(0.3,0.3,0.3,1);
	const float4 debugcolor = (float4)(0.4,0.4,0.4,1);
	
	// shading params
	const float4 diffuse = (float4)(1,0,0,1);
	const float4 specular = (float4)(1,1,1,1);
	const float specularity = 50.f;
	const float aostrength = 2.f;
	
	// volume dimensions
	const float3 size = D * convert_float3(M);
	const float3 invD = 1 / D;
	
	// raycasting aux data
	float dataValue = 0;
	float3 P, Pprev, step;
	float cellSize = fmin(fmin(D.x, D.y), D.z);
	float samplingRate = 3.5f;
	if (lin == 0) {
		samplingRate = 0.3f;
	}
	float dt = cellSize * samplingRate;
	float eps = 0.01f;
	
	#ifdef ADAPTIVE_SAMPLING
	const float k1 = 0.001f;
	const float k2 = 0.001f;
	const float k3 = 0.00005f;
	const float x0 = 100.f;
	const float y0 = 1.f;
	const float y1 = 3.f;
	#endif
	
	float2 result = (float2)(0,0); // raycasting lambda result buffer
	if (raybox(S, E, (float3)(0,0,0), size, &result) && result.y > 0) {
		result.x = fmax(result.x, 0); // start ray from camera, not from behind the camera
		
		float t = result.x + eps;
		
		
		
		#ifdef MIP
		float maxval = 0.f;
		while (t < result.y) {
			P = S + t * E;
			dataValue = sample(matrix, M, invD, P);
			if (dataValue < threshold) dataValue = 0;
			if (dataValue > maxval) { maxval = dataValue; }
			t += dt;
		}
		const float strength = 150.f; // todo cl arg
		color = lerp4(background, diffuse, maxval * strength);
		
		
		#else
		const float kappa = 1.f;
		float alpha = 0.f;
		*depth = result.x;
		while (t < result.y && color.w < 0.95f) {
			P = S + t * E;
			dataValue = sample(matrix, M, invD, P);
			
			#ifdef ADAPTIVE_SAMPLING
			dt = cellSize/(((exp(-k1*t*t) + exp(-k2*(t-x0)*(t-x0)))*(y1-y0)+y0)*exp(-k3*t*t));
			#endif

			float4 mapping = trf[clamp((int)(dataValue * (float) trfSamples), 0, trfSamples - 1)];
			//if (dataValue < threshold) dataValue = 0;
			
			alpha = 1.f - exp(-mapping.w * kappa * dt);
			currentColor = mapping * alpha;
			currentColor.w = alpha;
			color += (1.f - color.w) * currentColor;
			*depth += (1.f - color.w) * dt;
			
			t += dt;
		}
		
		color = lerp4(background, color, color.w);
		#endif
	
	
	
	
	} else {
		if (result.y < 0)
			color = background;
		else
			color = debugcolor;
	}
	
	return color;
}






///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Raycasting algorithm. Used in raycasting kernels with octree pointer.     //
//                                                                           //
// params:                                                                   //
//   S             - start of ray                                            //
//   E             - direction of ray                                        //
//   matrix        - input volume data                                       //
//   M             - matrix array dimensions                                 //
//   D             - volume dimensions                                       //
//   threshold     - isosurface value                                        //
//   lin           - generic debugging int                                   //
//   octree        - octree structure                                        //
//   octreeLevels  - number of octree levels                                 //
//   depth         - pointer to depth buffer                                 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
float4 castOctree(
float3 S, float3 E,
global const float* matrix,
const int3 M, const float3 D,
const float threshold, const int lin,
global const float* octree, const int octreeLevels,
float* depth
) {
	//return (float4)(0.3,0.6,0.9,1);
	float4 color = (float4)(0,0,0,0); // output color
	const float4 background = (float4)(0.3,0.3,0.3,1);
	const float4 debugcolor = (float4)(0.4,0.4,0.4,1);
	
	// shading params
	const float4 diffuse = (float4)(1,0,0,1);
	const float4 specular = (float4)(1,1,1,1);
	const float specularity = 50.f;
	const float aostrength = 2.f;
	
	// volume dimensions
	int octreeSize = 0;
	
	int Mmax = max(max(M.x, M.y), M.z);
	//const float3 size = D * convert_float3(M);
	const float3 size = D * pow(2, ceil(log2((float)Mmax)));
	const float3 invD = 1 / D;
	
	// raycasting aux data
	float dataValue = 0;
	float3 P, step;
	int octreeLevel = 0;
	
	float3 L = S;
	
	float3 boxmin = (float3)0;
	float3 boxmax = size;
	float3 half;

	float cellSize = fmin(fmin(D.x, D.y), D.z);
	//float3 eps = E * (cellSize * 0.5f);
	float eps = cellSize * 0.01f;

	// child index. 3*octreeIndex+{0,1,2} = min,max,avg
	int octreeIndex = 0;
	int iter = 0;
	
	int safetyCounter = 0;
	
	float2 result = (float2)(0,0); // raycasting lambda result buffer
	if (raybox(S, E, boxmin, boxmax, &result) && result.y > 0) {
		result.x = fmax(result.x, 0); // start ray from camera, not from behind the camera
		float maxLambda = result.y;
		
		P = S + (result.x + eps) * E;

		while (++safetyCounter < MAX_ITER) {
			float min = octree[3*octreeIndex  ];
			float max = octree[3*octreeIndex+1];
			float avg = octree[3*octreeIndex+2];

			if (threshold < min || threshold > max) {
				// advance / pop
				if (!raybox(S, E, boxmin, boxmax, &result)) {
					color = (float4)(1,0,1,1);
					break;
				}
				if (result.y + eps > maxLambda) {
					color = background;
					break; // out of root node
				}
				P = S + (result.y + eps) * E;

				// kd-restart
				octreeLevel = 0;
				boxmin = (float3)(0,0,0);
				boxmax = size;
				octreeIndex = 0;
			} else {
				if (octreeLevel == octreeLevels) {
					if (!raybox(S, E, boxmin, boxmax, &result)) {
						color = (float4)(0,1,0,1);
					} else {
						result.x = fmax(result.x, 0.f);
						
						P = S + (result.x + eps) * E;
						const float samplingDistance = 0.5f;
						const float delta = cellSize * samplingDistance;
						float t = result.x + eps;
						while (t <= result.y) {
							P = S + t * E;
							dataValue = sample(matrix, M, invD, P);
							
							if (dataValue > threshold) {
								// intersection point refinement
								float3 Pprev = S + (t - delta) * E;
								P = regulaFalsi(Pprev, P, matrix, M, invD, 5, threshold);
								*depth = fast_distance(S, P);
							
								// gradient estimate
								float3 Nv = fast_normalize(sampleNormal(matrix, M, invD, P));
								
								// shade
								float3 Lv = fast_normalize(L - P); // light vector
								float3 Lrv = (2.f * dot(Lv, Nv)) * Nv - Lv; // reflected light vector
								float3 Cv = -E; // camera vector
								float dk = clamp(dot(-Lv, Nv), 0.f, 1.f); // diffuse coef
								float sk = clamp(pow(fmax(dot(Cv, Lrv), 0.f), specularity), 0.f, 1.f); // specular coef
								//if (lin == 1 && *depth < cellSize * AO_DEPTH) {
									//float ao = ambientOcclusion(P, -Nv, matrix, M, invD, threshold, cellSize);
									//color = (diffuse * dk + specular * sk) * fmax(1.f - aostrength * ao, 0.f);
								//} else {
									color = diffuse * dk + specular * sk;
								//}
								break;
							}
							
							t += delta;
						}
						
						if (t > result.y) {
							// advance / pop
							if (!raybox(S, E, boxmin, boxmax, &result)) {
								color = (float4)(1,0,1,1);
								break;
							}
							if (result.y + eps > maxLambda) {
								color = background;
								break; // out of root node
							}
							P = S + (result.y + eps) * E;

							// kd-restart
							octreeLevel = 0;
							boxmin = (float3)(0,0,0);
							boxmax = size;
							octreeIndex = 0;
						} else {
							break;
						}
					}
					
					
					
					
				} else {
					octreeLevel++;
					//half = boxmin + (boxmax - boxmin) * 0.5f;
					half = (boxmin + boxmax) * 0.5f;
					int idx;
					if (P.x > half.x) {
						boxmin.x = half.x;
						if (P.y > half.y) {
							boxmin.y = half.y;
							if (P.z > half.z) {
								boxmin.z = half.z;
								idx = 8; 
							} else {
								boxmax.z = half.z;
								idx = 4;
							}
						} else {
							boxmax.y = half.y;
							if (P.z > half.z) {
								boxmin.z = half.z;
								idx = 6;
							} else {
								boxmax.z = half.z;
								idx = 2;
							}
						}
					} else {
						boxmax.x = half.x;
						if (P.y > half.y) {
							boxmin.y = half.y;
							if (P.z > half.z) {
								boxmin.z = half.z;
								idx = 7;
							} else {
								boxmax.z = half.z;
								idx = 3;
							}
						} else {
							boxmax.y = half.y;
							if (P.z > half.z) {
								boxmin.z = half.z;
								idx = 5;
							} else {
								boxmax.z = half.z;
								idx = 1;
							}
						}
					}
					octreeIndex = 8 * octreeIndex + idx;
					// DONE: find CHILD: set boxmin, boxmax, boxid, REPEAT :)
				}
			}
		} // end while


	
	} else {
		if (result.y < 0)
			color = background;
		else
			color = debugcolor;
	}
	
	// iteration count
	//float d = safetyCounter / (float) MAX_ITER;
	//color = (float4)(d,d,d,1);
	
	return color;
}







///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// The kernel used for raycasting. Fully raycasts every pixel.               //
//                                                                           //
// params:                                                                   //
//   output        - output image                                            //
//   input         - input image                                             //
//   outputDepth   - output depth image                                      //
//   inputDepth    - input depth image                                       //
//   Cx, Cy, Cz    - camera position                                         //
//   Ux, Uy, Uz    - camera unit up vector                                   //
//   Dx, Dy, Dz    - camera unit directional vector                          //
//   Rx, Ry, Rz    - camera unit right vector                                //
//   fov           - field of view = tan(angle)                              //
//   asr           - aspect ratio                                            //
//   matrix        - input volume data                                       //
//   mX, mY, mZ    - matrix array dimensions                                 //
//   dX, dY, dZ    - volume dimensions                                       //
//   octree        - octree structure                                        //
//   octreeLevels  - number of octree levels                                 //
//   threshold     - isosurface value                                        //
//   lin           - generic debugging int                                   //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
kernel void raycast(
OUTPUT_TYPE output, INPUT_TYPE input, OUTPUT_TYPE outputDepth, INPUT_TYPE inputDepth,
const float Cx, const float Cy, const float Cz,
const float Ux, const float Uy, const float Uz,
const float Dx, const float Dy, const float Dz,
const float Rx, const float Ry, const float Rz,
const float fov, const float asr,
global const float* matrix,
const int mX, const int mY, const int mZ,
const float dX, const float dY, const float dZ,
global const float* octree, const int octreeLevels,
global const float4* trf, const int trfSamples,
const float threshold, const int lin
) {
    unsigned int ix = get_global_id(0);
    unsigned int iy = get_global_id(1);
	unsigned int iw = get_global_size(0);
	unsigned int ih = get_global_size(1);
	float alphax = ix / (float) iw - 0.5f;
	float alphay = iy / (float) ih - 0.5f;
	
	float viewx = alphax * fov;
	float viewy = alphay * fov / asr;
	
	// ray direction
	float rx = Dx + viewx * Rx + viewy * Ux;
	float ry = Dy + viewx * Ry + viewy * Uy;
	float rz = Dz + viewx * Rz + viewy * Uz;
	
	// raycast
	float3 S = (float3)(Cx,Cy,Cz); // ray start
	float3 E = fast_normalize((float3)(rx,ry,rz)); // ray direction

	// volume
	int3 M = (int3)(mX, mY, mZ);
	float3 D = (float3)(dX, dY, dZ);
	
	float depth;
	float4 color = cast(S, E, matrix, M, D, threshold, lin, &depth);
	

	#ifdef USE_TEXTURE
		write_imagef(output, (int2)(ix, iy), color);
		writeDepth(outputDepth, (int2)(ix, iy), depth);
	#else
		writeColorUint(output, iy * iw + ix, color);
		outputDepth[iy * iw + ix] = as_uint(depth);
	#endif
}





///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// The kernel used for raycasting. Uses compositing for color synthesis.     //
//                                                                           //
// params:                                                                   //
//   output        - output image                                            //
//   input         - input image                                             //
//   outputDepth   - output depth image                                      //
//   inputDepth    - input depth image                                       //
//   Cx, Cy, Cz    - camera position                                         //
//   Ux, Uy, Uz    - camera unit up vector                                   //
//   Dx, Dy, Dz    - camera unit directional vector                          //
//   Rx, Ry, Rz    - camera unit right vector                                //
//   fov           - field of view = tan(angle)                              //
//   asr           - aspect ratio                                            //
//   matrix        - input volume data                                       //
//   mX, mY, mZ    - matrix array dimensions                                 //
//   dX, dY, dZ    - volume dimensions                                       //
//   octree        - octree structure                                        //
//   octreeLevels  - number of octree levels                                 //
//   threshold     - isosurface value                                        //
//   lin           - generic debugging int                                   //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
kernel void raycastAlpha(
OUTPUT_TYPE output, INPUT_TYPE input, OUTPUT_TYPE outputDepth, INPUT_TYPE inputDepth,
const float Cx, const float Cy, const float Cz,
const float Ux, const float Uy, const float Uz,
const float Dx, const float Dy, const float Dz,
const float Rx, const float Ry, const float Rz,
const float fov, const float asr,
global const float* matrix,
const int mX, const int mY, const int mZ,
const float dX, const float dY, const float dZ,
global const float* octree, const int octreeLevels,
global const float4* trf, const int trfSamples,
const float threshold, const int lin
) {
    unsigned int ix = get_global_id(0);
    unsigned int iy = get_global_id(1);
	unsigned int iw = get_global_size(0);
	unsigned int ih = get_global_size(1);
	float alphax = ix / (float) iw - 0.5f;
	float alphay = iy / (float) ih - 0.5f;
	
	float viewx = alphax * fov;
	float viewy = alphay * fov / asr;
	
	// ray direction
	float rx = Dx + viewx * Rx + viewy * Ux;
	float ry = Dy + viewx * Ry + viewy * Uy;
	float rz = Dz + viewx * Rz + viewy * Uz;
	
	// raycast
	float3 S = (float3)(Cx,Cy,Cz); // ray start
	float3 E = fast_normalize((float3)(rx,ry,rz)); // ray direction

	// volume
	int3 M = (int3)(mX, mY, mZ);
	float3 D = (float3)(dX, dY, dZ);
	
	float depth = FLT_MAX;
	float4 color = castAlpha(S, E, matrix, M, D, trf, trfSamples, threshold, lin, &depth);
	

	#ifdef USE_TEXTURE
		write_imagef(output, (int2)(ix, iy), color);
		writeDepth(outputDepth, (int2)(ix, iy), depth);
	#else
		writeColorUint(output, iy * iw + ix, color);
		outputDepth[iy * iw + ix] = as_uint(depth);
	#endif
}






///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// The kernel used for raycasting. Raycasts only odd pixels.                 //
//                                                                           //
// params:                                                                   //
//   output        - output image                                            //
//   input         - input image                                             //
//   outputDepth   - output depth image                                      //
//   inputDepth    - input depth image                                       //
//   Cx, Cy, Cz    - camera position                                         //
//   Ux, Uy, Uz    - camera unit up vector                                   //
//   Dx, Dy, Dz    - camera unit directional vector                          //
//   Rx, Ry, Rz    - camera unit right vector                                //
//   fov           - field of view = tan(angle)                              //
//   asr           - aspect ratio                                            //
//   matrix        - input volume data                                       //
//   mX, mY, mZ    - matrix array dimensions                                 //
//   dX, dY, dZ    - volume dimensions                                       //
//   octree        - octree structure                                        //
//   octreeLevels  - number of octree levels                                 //
//   threshold     - isosurface value                                        //
//   lin           - generic debugging int                                   //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
kernel void raycastPass1(
OUTPUT_TYPE output, INPUT_TYPE input, OUTPUT_TYPE outputDepth, INPUT_TYPE inputDepth,
const float Cx, const float Cy, const float Cz,
const float Ux, const float Uy, const float Uz,
const float Dx, const float Dy, const float Dz,
const float Rx, const float Ry, const float Rz,
const float fov, const float asr,
global const float* matrix,
const int mX, const int mY, const int mZ,
const float dX, const float dY, const float dZ,
global const float* octree, const int octreeLevels,
global const float4* trf, const int trfSamples,
const float threshold, const int lin
) {
    unsigned int ix = get_global_id(0);
    unsigned int iy = get_global_id(1);
	
	float4 color = (float4)(0,0,0,1);
	float depth = FLT_MAX;
	
	if (ix % 2 == 0 && iy % 2 == 0) { // sparse sampling
		unsigned int iw = get_global_size(0);
		unsigned int ih = get_global_size(1);
		float alphax = ix / (float) iw - 0.5f;
		float alphay = iy / (float) ih - 0.5f;
		
		float viewx = alphax * fov;
		float viewy = alphay * fov / asr;
		
		// ray direction
		float rx = Dx + viewx * Rx + viewy * Ux;
		float ry = Dy + viewx * Ry + viewy * Uy;
		float rz = Dz + viewx * Rz + viewy * Uz;
		
		// raycast
		float3 S = (float3)(Cx,Cy,Cz); // ray start
		float3 E = fast_normalize((float3)(rx,ry,rz)); // ray direction

		// volume
		int3 M = (int3)(mX, mY, mZ);
		float3 D = (float3)(dX, dY, dZ);
		
		//color = cast(S, E, matrix, M, D, threshold, lin, &depth);
		color = castOctree(S, E, matrix, M, D, threshold, lin, octree, octreeLevels, &depth);
		
	}
	

	#ifdef USE_TEXTURE
		write_imagef(output, (int2)(ix, iy), color);
		writeDepth(outputDepth, (int2)(ix, iy), depth);
	#else
		writeColorUint(output, iy * iw + ix, color);
		outputDepth[iy * iw + ix] = as_uint(depth);
	#endif
}






///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// The kernel used for raycasting. Extends first pass with interpolation.    //
//                                                                           //
// params:                                                                   //
//   output        - output image                                            //
//   input         - input image                                             //
//   outputDepth   - output depth image                                      //
//   inputDepth    - input depth image                                       //
//   Cx, Cy, Cz    - camera position                                         //
//   Ux, Uy, Uz    - camera unit up vector                                   //
//   Dx, Dy, Dz    - camera unit directional vector                          //
//   Rx, Ry, Rz    - camera unit right vector                                //
//   fov           - field of view = tan(angle)                              //
//   asr           - aspect ratio                                            //
//   matrix        - input volume data                                       //
//   mX, mY, mZ    - matrix array dimensions                                 //
//   dX, dY, dZ    - volume dimensions                                       //
//   octree        - octree structure                                        //
//   octreeLevels  - number of octree levels                                 //
//   threshold     - isosurface value                                        //
//   lin           - generic debugging int                                   //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
kernel void raycastPass2(
OUTPUT_TYPE output, INPUT_TYPE input, OUTPUT_TYPE outputDepth, INPUT_TYPE inputDepth,
const float Cx, const float Cy, const float Cz,
const float Ux, const float Uy, const float Uz,
const float Dx, const float Dy, const float Dz,
const float Rx, const float Ry, const float Rz,
const float fov, const float asr,
global const float* matrix,
const int mX, const int mY, const int mZ,
const float dX, const float dY, const float dZ,
global const float* octree, const int octreeLevels,
global const float4* trf, const int trfSamples,
const float threshold, const int lin
) {
	int ix = get_global_id(0);
    int iy = get_global_id(1);
	int iw = get_global_size(0);
	int ih = get_global_size(1);
	
	float4 color;
	float depth = FLT_MAX;
	if (ix % 2 == 0 && iy % 2 == 0) { // sparse sampling
		#ifdef USE_TEXTURE
			color = read_imagef(input, nearest_sampler, (int2)(ix, iy));
			depth = readDepth(inputDepth, (int2)(ix, iy));
		#else
			color = readColorUint(input, iy * iw + ix);
			depth = as_float(inputDepth[iy * iw + ix]);
		#endif
	} else {
		int dx = ix % 2;
		int dy = iy % 2;
		
		int ix0 = ix - dx;
		int ix1 = min(ix0 + 2, iw - 2);
		int iy0 = iy - dy;
		int iy1 = min(iy0 + 2, ih - 2);

		float4 c00, c01, c10, c11, c0, c1;
		float d00, d01, d10, d11, d0, d1;
		
		#ifdef USE_TEXTURE
			const sampler_t sampler = CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST;
			c00 = read_imagef(input, sampler, (int2)(ix0, iy0));
			c01 = read_imagef(input, sampler, (int2)(ix0, iy1));
			c10 = read_imagef(input, sampler, (int2)(ix1, iy0));
			c11 = read_imagef(input, sampler, (int2)(ix1, iy1));
			d00 = readDepth(inputDepth, (int2)(ix0, iy0));
			d01 = readDepth(inputDepth, (int2)(ix0, iy1));
			d10 = readDepth(inputDepth, (int2)(ix1, iy0));
			d11 = readDepth(inputDepth, (int2)(ix1, iy1));
		#else
			c00 = readColorUint(input, iy0 * iw + ix0);
			c01 = readColorUint(input, iy1 * iw + ix0);
			c10 = readColorUint(input, iy0 * iw + ix1);
			c11 = readColorUint(input, iy1 * iw + ix1);
			d00 = as_float(inputDepth[iy0 * iw + ix0]);
			d01 = as_float(inputDepth[iy1 * iw + ix0]);
			d10 = as_float(inputDepth[iy0 * iw + ix1]);
			d11 = as_float(inputDepth[iy1 * iw + ix1]);
		#endif
		
		// if gradient is too steep, cast a ray
		if (fmax(
			fmax(fast_distance(c00, c01), fast_distance(c10, c11)),
			fmax(fast_distance(c00, c10), fast_distance(c01, c11))) > SPARSE_SAMPLING_INTERPOLATION_THRESHOLD)
		{
			float alphax = ix / (float) iw - 0.5f;
			float alphay = iy / (float) ih - 0.5f;
			
			float viewx = alphax * fov;
			float viewy = alphay * fov / asr;
			
			// ray direction
			float rx = Dx + viewx * Rx + viewy * Ux;
			float ry = Dy + viewx * Ry + viewy * Uy;
			float rz = Dz + viewx * Rz + viewy * Uz;
			
			// raycast
			float3 S = (float3)(Cx,Cy,Cz); // ray start
			float3 E = fast_normalize((float3)(rx,ry,rz)); // ray direction

			// volume
			int3 M = (int3)(mX, mY, mZ);
			float3 D = (float3)(dX, dY, dZ);
			
			//color = cast(S, E, matrix, M, D, threshold, lin, &depth);
			color = castOctree(S, E, matrix, M, D, threshold, lin, octree, octreeLevels, &depth);
		} else {
			float alpha1 = (float) dy * 0.5f;
			float alpha2 = (float) dx * 0.5f;
			c0 = lerp4(c00, c01, alpha1);
			c1 = lerp4(c10, c11, alpha1);
			color = lerp4(c0, c1, alpha2);
			d0 = lerp(d00, d01, alpha1);
			d1 = lerp(d10, d11, alpha1);
			depth = lerp(d0, d1, alpha2);
		}
	}
	
	
	
	#ifdef USE_TEXTURE
		write_imagef(output, (int2)(ix, iy), color);
		writeDepth(outputDepth, (int2)(ix, iy), depth);
	#else
		writeColorUint(output, iy * iw + ix, color);
		outputDepth[iy * iw + ix] = as_uint(depth);
	#endif
}







///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// The kernel used for raycasting. Fully raycasts every pixel.               //
//                                                                           //
// params:                                                                   //
//   output        - output image                                            //
//   input         - input image                                             //
//   outputDepth   - output depth image                                      //
//   inputDepth    - input depth image                                       //
//   Cx, Cy, Cz    - camera position                                         //
//   Ux, Uy, Uz    - camera unit up vector                                   //
//   Dx, Dy, Dz    - camera unit directional vector                          //
//   Rx, Ry, Rz    - camera unit right vector                                //
//   fov           - field of view = tan(angle)                              //
//   asr           - aspect ratio                                            //
//   matrix        - input volume data                                       //
//   mX, mY, mZ    - matrix array dimensions                                 //
//   dX, dY, dZ    - volume dimensions                                       //
//   octree        - octree structure                                        //
//   octreeLevels  - number of octree levels                                 //
//   threshold     - isosurface value                                        //
//   lin           - generic debugging int                                   //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
kernel void raycastOctree(
OUTPUT_TYPE output, INPUT_TYPE input, OUTPUT_TYPE outputDepth, INPUT_TYPE inputDepth,
const float Cx, const float Cy, const float Cz,
const float Ux, const float Uy, const float Uz,
const float Dx, const float Dy, const float Dz,
const float Rx, const float Ry, const float Rz,
const float fov, const float asr,
global const float* matrix,
const int mX, const int mY, const int mZ,
const float dX, const float dY, const float dZ,
global const float* octree, const int octreeLevels,
global const float4* trf, const int trfSamples,
const float threshold, const int lin
) {
    unsigned int ix = get_global_id(0);
    unsigned int iy = get_global_id(1);
	unsigned int iw = get_global_size(0);
	unsigned int ih = get_global_size(1);
	float alphax = ix / (float) iw - 0.5f;
	float alphay = iy / (float) ih - 0.5f;
	
	float viewx = alphax * fov;
	float viewy = alphay * fov / asr;
	
	// ray direction
	float rx = Dx + viewx * Rx + viewy * Ux;
	float ry = Dy + viewx * Ry + viewy * Uy;
	float rz = Dz + viewx * Rz + viewy * Uz;
	
	// raycast
	float3 S = (float3)(Cx,Cy,Cz); // ray start
	float3 E = fast_normalize((float3)(rx,ry,rz)); // ray direction

	// volume
	int3 M = (int3)(mX, mY, mZ);
	float3 D = (float3)(dX, dY, dZ);
	
	float depth = FLT_MAX;
	float4 color = castOctree(S, E, matrix, M, D, threshold, lin, octree, octreeLevels, &depth);
	

	#ifdef USE_TEXTURE
		write_imagef(output, (int2)(ix, iy), color);
		writeDepth(outputDepth, (int2)(ix, iy), depth);
	#else
		writeColorUint(output, iy * iw + ix, color);
		outputDepth[iy * iw + ix] = as_uint(depth);
	#endif
}







///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// The kernel used for depth of field effect.                                //
//                                                                           //
// params:                                                                   //
//   clear        - clear image                                              //
//   blurred      - blurred image                                            //
//   depthBuffer  - depth image                                              //
//   output       - output image                                             //
//   focus        - focus distance                                           //
//   strength     - effect strength                                          //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
kernel void depthOfField(INPUT_TYPE clear, INPUT_TYPE blurred, INPUT_TYPE depthBuffer, OUTPUT_TYPE output, const float focus, const float strength) {
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	float4 cclear, cblurred, color;
	float depth;

	#ifdef USE_TEXTURE
		cclear = read_imagef(clear, nearest_sampler, (int2)(x, y));
		cblurred = read_imagef(blurred, nearest_sampler, (int2)(x, y));
		depth = readDepth(depthBuffer, (int2)(x, y));
	#else
		int idx = x + get_global_size(0) * y;
		// TODO read all
	#endif

	const float dist = fabs(focus - depth);
	const float alpha = clamp(dist * strength, 0.f, 1.f);
	color = lerp4(cclear, cblurred, alpha);
	//if (depth != FLT_MAX)
		color *= (1.f - alpha) * 0.6f + 0.4f;
	//float d = depth / 500.f;
	//color = (float4)(d,d,d,1);

	#ifdef USE_TEXTURE
		write_imagef(output, (int2)(x, y), color);
	#else
		writeColorUint(output, iy * iw + ix, color);
	#endif
}








///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// The kernel used for screen-space ambient occlusion (SSAO) effect.         //
//                                                                           //
// params:                                                                   //
//   image        - input rendered image                                     //
//   depthBuffer  - depth image                                              //
//   output       - output image                                             //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
kernel void ssao(INPUT_TYPE image, INPUT_TYPE depthBuffer, OUTPUT_TYPE output) {
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int w = get_global_size(0);
	const int h = get_global_size(1);

	float4 color;
	float depth[SSAO_SAMPLES];
	float thisDepth;
	float2 pos = (float2)((float)x, (float)y);
	
	const float2 ssao_pos[SSAO_SAMPLES] = {
		(float2)( 0.1905,   -0.3775),
		(float2)( 0.0937,   -0.0102),
		(float2)(-0.0741,   -0.5939),
		(float2)(-0.2619,    0.3914),
		(float2)(-0.6011,   -0.3508),
		(float2)( 0.5455,    0.4385),
		(float2)( 0.5310,   -0.3546),
		(float2)( 0.0244,   -0.0231),
		(float2)( 0.0284,   -0.0627),
		(float2)(-0.0215,    0.3189),
		(float2)(-0.4403,   -0.2966),
		(float2)( 0.6479,    0.0923),
		(float2)(-0.3635,    0.1845),
		(float2)(-0.3148,    0.7571),
		(float2)( 0.3792,    0.6101),
		(float2)( 0.4192,    0.8732)
	};
	const float kappa = 0.01f;
	const float ssaoSize = 1000.f;
	float sampleSize;
	
	
	#ifdef USE_TEXTURE
		// random rotation
		float phi = x * x  * y + y * y;
		float sphi = sin(phi);
		float cphi = cos(phi);
		// kernel rotation without blur is ugly
		//float2 rota = (float2)(cphi, sphi);
		//float2 rotb = (float2)(-sphi, cphi);
		
		color = read_imagef(image, nearest_sampler, convert_int2(pos));
		thisDepth = readDepth(depthBuffer, convert_int2(pos));
		float invThisDepth = 1.f / thisDepth;
		for (int i=0; i<SSAO_SAMPLES; i++) {
			//float2 samplepos = ssao_pos[i].x * rota + ssao_pos[i].y * rotb;
			float2 samplepos = ssao_pos[i];
			
			sampleSize = ssaoSize * invThisDepth;
			int2 readpos = convert_int2(pos + samplepos * sampleSize);// * exp(-thisDepth * kappa));
			readpos = clamp(readpos, (int2)(0, 0), (int2)(w - 1, h - 1));
			depth[i] = readDepth(depthBuffer, readpos);
		}
	#else
		int idx = x + get_global_size(0) * y;
		// TODO read all
	#endif
	
	float inv = 1.f / SSAO_SAMPLES;
	float s = 0;
	for (int i=0; i<SSAO_SAMPLES; i++) {
		if (thisDepth > depth[i] && thisDepth - depth[i] > 10.f && thisDepth - depth[i] < 100.f) {
			s += inv;
		}
	}
	
	color *= 1 - s;
	
	
	
	#ifdef USE_TEXTURE
		write_imagef(output, (int2)(x, y), color);
	#else
		writeColorUint(output, y * w + x, color);
	#endif
}












