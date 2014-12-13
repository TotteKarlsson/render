/**
 * License: GPL
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License 2
 * as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */
package org.janelia.alignment;

import ij.ImagePlus;
import ij.process.ImageProcessor;
import ij.IJ;

import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Writer;
import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.lang.StringBuilder;

import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.lang.Math;
import java.awt.Rectangle;

import mpicbg.imagefeatures.Feature;
import mpicbg.imagefeatures.FloatArray2DSIFT;
import mpicbg.ij.SIFT;
import mpicbg.models.CoordinateTransform;
import mpicbg.models.CoordinateTransformList;
import mpicbg.models.InterpolatedCoordinateTransform;
import mpicbg.models.AffineModel2D;
import mpicbg.models.TranslationModel2D;
import mpicbg.models.InvertibleBoundable;

import mpicbg.models.CoordinateTransformMesh;
import mpicbg.ij.TransformMeshMapping;
import ij.process.ImageStatistics;
import ij.measure.Calibration;
import ij.measure.Measurements;
import ij.plugin.ContrastEnhancer;
import ij.io.FileSaver;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonSyntaxException;


import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.io.ImgIOException;
import net.imglib2.io.ImgOpener;


//import mpicbg.imglib.type.numeric.real.FloatType;
//import mpicbg.imglib.type.numeric.integer.UnsignedShortType;
//import mpicbg.imglib.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.integer.UnsignedByteType;

import mpicbg.stitching.fusion.Fusion;
import mpicbg.stitching.ImageCollectionElement;
import mpicbg.stitching.StitchingParameters;
import mpicbg.stitching.ImagePlusTimePoint;
import mpicbg.stitching.CollectionStitchingImgLib;


/**
 * 
 * @author Seymour Knowles-Barley
 */
public class StitchImagesByCC
{
	@Parameters
	static private class Params
	{
	  
    
		@Parameter( names = "--help", description = "Display this note", help = true )
        private final boolean help = false;

        @Parameter( names = "--imageFiles", description = "path to image files", required = true )
        private List<String> files;
        
        @Parameter( names = "--xoffsets", description = "x offsets", required = true )
        private List<Float> xoffsets;
        
         @Parameter( names = "--yoffsets", description = "y offsets", required = true )
        private List<Float> yoffsets;
        
        @Parameter( names = "--outputLayout", description = "path to save the layout file for these frames", required = false )
        private String outputLayout = null;
        
        @Parameter( names = "--outputFile", description = "Path to save image file", required = false )
        public String outputFile = null;
       

	}
	
	private StitchImagesByCC() {}
/*	
	private static float[] translateBoundingBox( float[] originalBoundingBox, CoordinateTransformList< CoordinateTransform > ctlMipmap )
	{
		float[][] points = {
				{ originalBoundingBox[0], originalBoundingBox[2] },
				{ originalBoundingBox[0], originalBoundingBox[3] },
				{ originalBoundingBox[0], originalBoundingBox[2] },
		};
	
	}
*/


	
	public static ArrayList<ImagePlus> read_images(List<String> filepaths)
	{
	    ArrayList<ImagePlus> images = new ArrayList();
	    
	    for ( final String filepath : filepaths )
	    {
	      System.out.println( filepath );
	     
	        ImagePlus img = IJ.openImage(filepath);
	     //try{
	     //     final Img< UnsignedByteType > img = new ImgOpener().openImg(filepath,
	     //     new ArrayImgFactory< UnsignedByteType >(), new UnsignedByteType() );
                
	      
	         images.add(img);
	      //}
	     //    catch ( final ImgIOException e )
        //{
        //    e.printStackTrace();
      //  }
	    }
	    return images;
	    
	}
	public static void main( final String[] args )
	{
		
		final Params params = new Params();
		try
        {
			final JCommander jc = new JCommander( params, args );
        	if ( params.help )
            {
        		jc.usage();
                return;
            }
        }
        catch ( final Exception e )
        {
        	e.printStackTrace();
            final JCommander jc = new JCommander( params );
        	jc.setProgramName( "java [-options] -cp render.jar org.janelia.alignment.ReconstructImage" );
        	jc.usage(); 
        	return;
        }
        

      ArrayList<ImageCollectionElement> elements=new ArrayList();
      
      for(int i = 0 ; i<params.files.size();i++){
        
        File file = new File(params.files.get(i));
        System.out.println( file.getAbsolutePath());
        ImageCollectionElement element=new ImageCollectionElement(file,i);
        element.setDimensionality( 2 );
        float[] offsets = {params.xoffsets.get(i),params.yoffsets.get(i)};
        element.setOffset(offsets);
        element.setModel(new TranslationModel2D() );
        elements.add(element);
      }
  
        StitchingParameters stitchparams=new StitchingParameters();
        stitchparams.dimensionality = 2;
        stitchparams.channel1 = 0;
        stitchparams.channel2 = 0;
        stitchparams.timeSelect = 0;
        stitchparams.checkPeaks = 5;
        stitchparams.regThreshold = 0.7f;
        stitchparams.computeOverlap = true;
        stitchparams.subpixelAccuracy = true;
        stitchparams.fusionMethod = 0;
  
      
      
      ArrayList <ImagePlusTimePoint> optimized = CollectionStitchingImgLib.stitchCollection( elements, stitchparams );
	    
	       //read in the images
	    final ArrayList<ImagePlus> images = read_images(params.files);
	    
	    ArrayList <InvertibleBoundable> models = new ArrayList();

      for  (ImagePlusTimePoint imt : optimized){
        models.add((InvertibleBoundable) imt.getModel());
      }  
         
			

			 if (params.outputFile != null){
			   //stitch these images
  			  ImagePlus imp = Fusion.fuse(new UnsignedShortType(), images, models, 2, true, 0,null, false, false, false);
          FileSaver fs = new FileSaver( imp );
          fs.saveAsTiff(params.outputFile);
			 }
      if (params.outputLayout != null){
        Utils.writeObjectToFile(models,params.outputLayout);
      }
      
      
	    
	}
}
