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

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class MHDReader {
	public static int Nx;
	public static int Ny;
	public static int Nz;

	public static double dx;
	public static double dy;
	public static double dz;

	public static double tx;
	public static double ty;
	public static double tz;

	public static double[][] rotationMatrix = new double[4][4];
	public static boolean elementByteOrder;
	public static String elementType;

	public static String rawFile;

	public static void readMHD(String fileName) {
		File file = new File(fileName);
		Scanner sc = null;
		try {
			sc = new Scanner(file);
			String value = "";
			while (sc.hasNext()) {
				value = sc.next();
				sc.next(); // move past sign =
				if (value.equals("DimSize")) {
					Nx = sc.nextInt();
					Ny = sc.nextInt();
					Nz = sc.nextInt();
				} else if (value.equals("ElementSpacing")) {
					dx = Double.parseDouble(sc.next());
					dy = Double.parseDouble(sc.next());
					dz = Double.parseDouble(sc.next());
				} else if (value.equals("Position")) {
					tx = sc.nextDouble();
					ty = sc.nextDouble();
					tz = sc.nextDouble();
				} else if (value.equals("Orientation")) {
					for (int i = 0; i < rotationMatrix.length; i++) {
						for (int j = 0; j < rotationMatrix[i].length; j++) {
							if (i >= 3 || j >= 3)
								rotationMatrix[i][j] = (i == j) ? 1 : 0;
							else
								rotationMatrix[i][j] = Double.parseDouble(sc.next());
						}
					}
				} else if (value.equals("AnatomicalOrientation")) {
					sc.nextLine();
				} else if (value.equals("ElementByteOrderMSB")) {
					elementByteOrder = Boolean.parseBoolean(sc.next().toLowerCase());
				} else if (value.equals("ElementType")) {
					elementType = sc.next();
				} else if (value.equals("ElementDataFile")) {
					rawFile = file.getParentFile().getAbsolutePath() + "\\" + sc.next();
				}
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} finally {
			if (sc != null)
				sc.close();
		}

	}

}
