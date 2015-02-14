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
import java.lang.reflect.Type;

import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.lang.Math;
import java.awt.Rectangle;

import mpicbg.imagefeatures.Feature;
import mpicbg.imagefeatures.FloatArray2DSIFT;
import mpicbg.ij.SIFT;
import mpicbg.ij.InverseTransformMapping;
import mpicbg.models.CoordinateTransform;
import mpicbg.models.CoordinateTransformList;
import mpicbg.models.InterpolatedCoordinateTransform;
import mpicbg.models.AffineModel2D;
import mpicbg.models.TranslationModel2D;
import mpicbg.models.RigidModel2D;
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
import com.google.gson.reflect.TypeToken;

//import com.thoughtworks.xstream.XStream;
import java.io.ObjectInputStream;

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

/**
 * 
 * @author Seymour Knowles-Barley
 */
public class ApplyStitching
{
	@Parameters
	static private class Params
	{
	  
    
		@Parameter( names = "--help", description = "Display this note", help = true )
        private final boolean help = false;

        @Parameter( names = "--imageFiles", description = "path to image files", required = true )
        private List<String> files;
        
        @Parameter( names = "--layoutFile", description = "layout file for these frames", required = true )
        private String layoutFile = null;
        
        
        @Parameter( names = "--outputFile", description = "Path to save image file", required = true )
        public String outputFile = null;
        

        

	}
	
	private ApplyStitching() {}
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

	
	public static ArrayList<InvertibleBoundable> read_layout_file(String filepath)
	{
	   try{
          ArrayList<InvertibleBoundable> models = new ArrayList();
          File file = new File(filepath);
          BufferedReader br = new BufferedReader(new FileReader(file));
          
          for (String line = br.readLine(); line != null; line = br.readLine()) 
          {
                String[] parts = line.split("\\s+");
                AffineModel2D model = new AffineModel2D();
                float[] flt_parts = new float[parts.length];
                for (int i=0;i<flt_parts.length;i++)
                {
                    flt_parts[i]=Float.parseFloat(parts[i].trim());
                }
          
          model.set(flt_parts[1],flt_parts[2],flt_parts[4],flt_parts[5],flt_parts[3],flt_parts[6]);
          models.add((InvertibleBoundable) model);
          } 
          return models;
	   }
    catch (final FileNotFoundException e){
          e.printStackTrace();
          return null;
      }
      catch (final IOException e ){
          e.printStackTrace();
          return null;
      }
	  
	  
	}
	
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
		
	    //read in the layouts
	    final ArrayList<TranslationModel2D> translation_models;
	    
	    Type listType = new TypeToken<ArrayList<TranslationModel2D>>() {}.getType();
	    
	    try {
	      final Gson gson = new Gson();
	      translation_models = gson.fromJson( new FileReader( params.layoutFile ), listType);
	    }
	    catch ( final Exception e )
      {
              e.printStackTrace( System.err );
              return;
      }
	    
		//convert to InvertibleBoundable
		final ArrayList<InvertibleBoundable> models = new ArrayList();
		for (final TranslationModel2D trans_model:translation_models){
		models.add((InvertibleBoundable) trans_model);
		}
		//read in the images
		final ArrayList<ImagePlus> images = read_images(params.files);

		//make sure each image has a model
		assert images.size() == models.size();

		//stitch these images
		ImagePlus imp = Fusion.fuse(new UnsignedShortType(), images, models, 2, true, 0,null, false, false, false);

	    	
		if (params.outputFile != null){
			//stitch these images
			FileSaver fs = new FileSaver( imp );
			File file = new File(params.outputFile);
			file.getParentFile().mkdirs();
			fs.saveAsTiff(params.outputFile);
		}
      
	    
	}
}
