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

package si.uni_lj.fri.lwjgltest;

public class OctreeNode {
	public float min = Float.MAX_VALUE;
	public float max = -Float.MAX_VALUE;
	public float avg = 0;
	public boolean isLeaf;
	public OctreeNode[] children = new OctreeNode[8];
	
	public OctreeNode(float[] matrix, int Nx, int Ny, int Nz, int x0, int y0, int z0, int x1, int y1, int z1, int level) {
		isLeaf = level == 0;
		if (isLeaf) {
			calcStats(matrix, Nx, Ny, Nz, x0, y0, z0, x1, y1, z1);
		} else {
			int d = (x1 - x0) / 2; // dx = dy = dz, afaik
			children[0] = new OctreeNode(matrix, Nx, Ny, Nz, x0, y0, z0, x0 + d, y0 + d, z0 + d, level - 1);
			children[1] = new OctreeNode(matrix, Nx, Ny, Nz, x0 + d, y0, z0, x1, y0 + d, z0 + d, level - 1);
			children[2] = new OctreeNode(matrix, Nx, Ny, Nz, x0, y0 + d, z0, x0 + d, y1, z0 + d, level - 1);
			children[3] = new OctreeNode(matrix, Nx, Ny, Nz, x0 + d, y0 + d, z0, x1, y1, z0 + d, level - 1);
			children[4] = new OctreeNode(matrix, Nx, Ny, Nz, x0, y0, z0 + d, x0 + d, y0 + d, z1, level - 1);
			children[5] = new OctreeNode(matrix, Nx, Ny, Nz, x0 + d, y0, z0 + d, x1, y0 + d, z1, level - 1);
			children[6] = new OctreeNode(matrix, Nx, Ny, Nz, x0, y0 + d, z0 + d, x0 + d, y1, z1, level - 1);
			children[7] = new OctreeNode(matrix, Nx, Ny, Nz, x0 + d, y0 + d, z0 + d, x1, y1, z1, level - 1);
			calcStats();
		}
	}
	
	public void calcStats() {
		avg = 0;
		for (OctreeNode n : children) {
			min = Math.min(min, n.min);
			max = Math.max(max, n.max);
			avg += n.avg / 8.f;
		}
	}
	
	public void calcStats(float[] matrix, int Nx, int Ny, int Nz, int x0, int y0, int z0, int x1, int y1, int z1) {
		float in = 1.f / (float) ((x1 - x0) * (y1 - y0) * (z1 - z0)); // inverse volume
		for (int i=x0; i<x1; i++) {
			for (int j=y0; j<y1; j++) {
				for (int k=z0; k<z1; k++) {
					float v = 0;
					if (i >= 0 && i < Nx && j >= 0 && j < Ny && k >= 0 && k < Nz) {
						v = matrix[i + j*Nx + k*Nx*Ny];
					}
					if (v < min) min = v;
					if (v > max) max = v;
					avg += v * in;
				}
			}
		}
	}
	
	public void storeInto(float[] buffer, int i) {
		buffer[3*i  ] = min;
		buffer[3*i+1] = max;
		buffer[3*i+2] = avg;
		if (!isLeaf) {
			// loop unrolled for speed
			int idx = 8 * i;
			children[0].storeInto(buffer, idx + 1);
			children[1].storeInto(buffer, idx + 2);
			children[2].storeInto(buffer, idx + 3);
			children[3].storeInto(buffer, idx + 4);
			children[4].storeInto(buffer, idx + 5);
			children[5].storeInto(buffer, idx + 6);
			children[6].storeInto(buffer, idx + 7);
			children[7].storeInto(buffer, idx + 8);
		}
	}
}
