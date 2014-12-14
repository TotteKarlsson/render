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
import ij.io.Opener;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.awt.image.RenderedImage;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.FileNotFoundException;
import java.net.URL;
import java.util.ArrayList;

import javax.imageio.IIOImage;
import javax.imageio.ImageIO;
import javax.imageio.ImageWriteParam;
import javax.imageio.ImageWriter;
import javax.imageio.stream.FileImageOutputStream;

import mpicbg.trakem2.transform.TranslationModel2D;
import mpicbg.trakem2.transform.RigidModel2D;
import mpicbg.trakem2.transform.SimilarityModel2D;
import mpicbg.trakem2.transform.AffineModel2D;
import mpicbg.trakem2.transform.HomographyModel2D;

import mpicbg.models.AbstractModel;
import mpicbg.models.CoordinateTransform;
import mpicbg.models.InterpolatedAffineModel2D;
import mpicbg.models.NotEnoughDataPointsException;
import mpicbg.models.Point;
import mpicbg.models.PointMatch;
import mpicbg.models.SpringMesh;
import mpicbg.models.Tile;

import java.io.Writer;
import java.io.FileWriter;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonSyntaxException;
import com.google.gson.reflect.TypeToken;

import com.thoughtworks.xstream.XStream;


/**
 * 
 *
 * @author Stephan Saalfeld <saalfeld@janelia.hhmi.org>, Seymour Knowles-Barley
 */
public class Utils
{
	private Utils() {}
	
	final static public class Triple< A, B, C >
	{
		final public A a;
		final public B b;
		final public C c;

		Triple( final A a, final B b, final C c )
		{
			this.a = a;
			this.b = b;
			this.c = c;
		}
	}
	
	/**
	 * Writes min(a,b) into a
	 * 
	 * @param a
	 * @param b
	 */
	final static public void min( final float[] a, final float[] b )
	{
		for ( int i = 0; i < a.length; ++i )
			if ( b[ i ] < a[ i ] ) a[ i ] = b[ i ];
	}

	/**
	 * Writes max(a,b) into a
	 * 
	 * @param a
	 * @param b
	 */
	final static public void max( final float[] a, final float[] b )
	{
		for ( int i = 0; i < a.length; ++i )
			if ( b[ i ] > a[ i ] ) a[ i ] = b[ i ];
	}
	
	/**
	 * Get a model from an integer specifier
	 */
	final static public AbstractModel< ? > createModel( final int modelIndex )
	{
		switch ( modelIndex )
		{
		case 0:
			return new TranslationModel2D();
		case 1:
			return new RigidModel2D();
		case 2:
			return new SimilarityModel2D();
		case 3:
			return new AffineModel2D();
		case 4:
			return new HomographyModel2D();
		default:
			return null;
		}
	}
	
	/**
	 * Get a tile from an integer specifier
	 */
	@SuppressWarnings({ "rawtypes", "unchecked" })
	final static public Tile createTile( final int modelIndex )
	{
		switch ( modelIndex )
		{
		case 0:
			return new Tile( new TranslationModel2D() );
		case 1:
			return new Tile( new RigidModel2D() );
		case 2:
			return new Tile( new SimilarityModel2D() );
		case 3:
			return new Tile( new AffineModel2D() );
		case 4:
			return new Tile( new HomographyModel2D() );
		default:
			return null;
		}
	}
	
	/**
	 * Get an interpolated affine tile from an integer specifier
	 */
	@SuppressWarnings({ "rawtypes", "unchecked" })
	final static public Tile createInterpolatedAffineTile( final int modelIndex, final int regularizerIndex, float lambda )
	{
		final AbstractModel< ? > m = createModel(modelIndex);
		final AbstractModel< ? > r = createModel(regularizerIndex);
		return new Tile( new InterpolatedAffineModel2D( m, r, lambda ) );
	}
	
	/**
	 * Generate a spring mesh from image dimensions and spring mesh parameters.
	 */
	public static SpringMesh getMesh( int imWidth, int imHeight, float layerScale,
			int resolutionSpringMesh, float stiffnessSpringMesh, float dampSpringMesh, float maxStretchSpringMesh )
	{
		final int meshWidth = ( int )Math.ceil( imWidth * layerScale );
		final int meshHeight = ( int )Math.ceil( imHeight * layerScale );
		
		final SpringMesh mesh = new SpringMesh(
						resolutionSpringMesh,
						meshWidth,
						meshHeight,
						stiffnessSpringMesh,
						maxStretchSpringMesh * layerScale,
						dampSpringMesh );
		System.out.println( "Generated mesh with " + mesh.numVertices() + " vertices.");
		return mesh;
	}

	final static private double LOG2 = Math.log( 2.0 );
	
	/**
	 * Save an image using ImageIO.
	 * 
	 * @param image
	 * @param path
	 * @param format
	 * @param quality
	 * 
	 * @return
	 */
	final static public boolean saveImage(
			final RenderedImage image,
			final String path,
			final String format,
			final float quality )
	{
		final File file = new File( path );
		final FileImageOutputStream output;
		try
		{
			file.getParentFile().mkdirs();
			final ImageWriter writer = ImageIO.getImageWritersByFormatName( format ).next();
			output = new FileImageOutputStream( file );
			writer.setOutput( output );
			if ( format.equalsIgnoreCase( "jpg" ) )
			{
				final ImageWriteParam param = writer.getDefaultWriteParam();
				param.setCompressionMode( ImageWriteParam.MODE_EXPLICIT );
				param.setCompressionQuality( quality );
				writer.write( null, new IIOImage( image, null, null ), param );
			}
			else
				writer.write( image );
			
			writer.dispose();
			output.close();
			
			return true;
		}
		catch ( final IOException e )
		{
			e.printStackTrace( System.err );
			return false;
		}
	}
	
	
	/**
	 * Save an image using ImageIO.
	 * 
	 * @param image
	 * @param path
	 * @param format
	 */
	final static public boolean saveImage(
			final RenderedImage image,
			final String path,
			final String format )
	{
		return saveImage( image, path, format, 0.85f );
	}
	
	/**
	 * Open an ImagePlus from a file.
	 * 
	 * @param pathString
	 * @return
	 */
	final static public ImagePlus openImagePlus( final String pathString )
	{
		final ImagePlus imp = new Opener().openImage( pathString );
		return imp;
	}
	
	/**int scaleLevel = 0;
		while ( invScale > 1 )
		{
			invScale >>= 1;
			++scaleLevel;
		}
		return scaleLevel
	 * Open an ImagePlus from a URL
	 * 
	 * @param urlString
	 * @return
	 */
	final static public ImagePlus openImagePlusUrl( final String urlString )
	{
		final ImagePlus imp = new Opener().openURL( imageJUrl( urlString ) );
		return imp;
	}
	
	/**
	 * Open an Image from a URL.  Try ImageIO first, then ImageJ.
	 * 
	 * @param urlString
	 * @return
	 */
	final static public BufferedImage openImageUrl( final String urlString )
	{
		BufferedImage image;
		try
		{
			final URL url = new URL( urlString );
			final BufferedImage imageTemp = ImageIO.read( url );
			
			/* This gymnastic is necessary to get reproducible gray
			 * values, just opening a JPG or PNG, even when saved by
			 * ImageIO, and grabbing its pixels results in gray values
			 * with a non-matching gamma transfer function, I cannot tell
			 * why... */
		    image = new BufferedImage( imageTemp.getWidth(), imageTemp.getHeight(), BufferedImage.TYPE_INT_ARGB );
			image.createGraphics().drawImage( imageTemp, 0, 0, null );
		}
		catch ( final Exception e )
		{
			try
			{
				final ImagePlus imp = openImagePlusUrl( urlString );
				if ( imp != null )
				{
					image = imp.getBufferedImage();
				}
				else image = null;
			}
			catch ( final Exception f )
			{
				image = null;
			}
		}
		return image;
	}
	
	
	/**
	 * Open an Image from a file.  Try ImageIO first, then ImageJ.
	 * 
	 * @param urlString
	 * @return
	 */
	final static public BufferedImage openImage( final String path )
	{
		BufferedImage image = null;
		try
		{
			final File file = new File( path );
			if ( file.exists() )
			{
				final BufferedImage jpg = ImageIO.read( file );
				
				/* This gymnastic is necessary to get reproducible gray
				 * values, just opening a JPG or PNG, even when saved by
				 * ImageIO, and grabbing its pixels results in gray values
				 * with a non-matching gamma transfer function, I cannot tell
				 * why... */
			    image = new BufferedImage( jpg.getWidth(), jpg.getHeight(), BufferedImage.TYPE_INT_ARGB );
				image.createGraphics().drawImage( jpg, 0, 0, null );
			}
		}
		catch ( final Exception e )
		{
			try
			{
				final ImagePlus imp = openImagePlus( path );
				if ( imp != null )
				{
					image = imp.getBufferedImage();
				}
				else image = null;
			}
			catch ( final Exception f )
			{
				image = null;
			}
		}
		return image;
	}
	
	
	/**
	 * If a URL starts with "file:", replace "file:" with "" because ImageJ wouldn't understand it otherwise
	 * @return
	 */
	final static private String imageJUrl( final String urlString )
	{
		return urlString.replace( "^file:", "" );
	}
	
	
	/**
	 * Combine a 0x??rgb int[] raster and an unsigned byte[] alpha channel into
	 * a 0xargb int[] raster.  The operation is perfomed in place on the int[]
	 * raster.
	 */
	final static public void combineARGB( final int[] rgb, final byte[] a )
	{
		for ( int i = 0; i < rgb.length; ++i )
		{
			rgb[ i ] &= 0x00ffffff;
			rgb[ i ] |= ( a[ i ] & 0xff ) << 24;
		}
	}
	
	
	/**
	 * Sample the average scaling of a given {@link CoordinateTransform} by transferring
	 * a set of point samples using the {@link CoordinateTransform} and then
	 * least-squares fitting a {@link SimilarityModel2D} to it.
	 * 
	 * @param ct
	 * @param width of the samples set
	 * @param height of the samples set
	 * @param dx spacing between samples
	 * 
	 * @return average scale factor
	 */
	final static public double sampleAverageScale( final CoordinateTransform ct, final int width, final int height, final double dx )
	{
		final ArrayList< PointMatch > samples = new ArrayList< PointMatch >();
		for ( float y = 0; y < height; y += dx )
		{
			for ( float x = 0; x < width; x += dx )
			{
				final Point p = new Point( new float[]{ x, y } );
				p.apply( ct );
				samples.add( new PointMatch( p, p ) );
			}
		}
		final SimilarityModel2D model = new SimilarityModel2D();
		try
		{
			model.fit( samples );
		}
		catch ( final NotEnoughDataPointsException e )
		{
			e.printStackTrace( System.err );
			return 1;
		}
		final double[] data = new double[ 6 ];
		model.toArray( data );
//		return 1;
		return Math.sqrt( data[ 0 ] * data[ 0 ] + data[ 1 ] * data[ 1 ] );
	}
	
	
	final static public int bestMipmapLevel( final double scale )
	{
		int invScale = ( int )( 1.0 / scale );
		int scaleLevel = 0;
		while ( invScale > 1 )
		{
			invScale >>= 1;
			++scaleLevel;
		}
		return scaleLevel;
	}
	
	
	/**
	 * Returns the exact fractional `index' of the desired scale in a power of
	 * 2 mipmap pyramid.
	 * 
	 * @param scale
	 * @return
	 */
	final static public double mipmapLevel( final double scale )
	{
		return Math.log( 1.0 / scale ) / LOG2;
	}
	
	
	/**
	 * Create an affine transformation that compensates for both scale and
	 * pixel shift of a mipmap level that was generated by top-left pixel
	 * averaging.
	 * 
	 * @param scaleLevel
	 * @return
	 */
	final static AffineModel2D createScaleLevelTransform( final int scaleLevel )
	{
		final AffineModel2D a = new AffineModel2D();
		final int scale = 1 << scaleLevel;
		final float t = ( scale - 1 ) * 0.5f;
		a.set( scale, 0, 0, scale, t, t );
		return a;
	}
	
	/**
	 * Create an affine transformation that compensates for both scale and
	 * pixel shift of a mipmap level that was generated by top-left pixel
	 * averaging.
	 * 
	 * @param scaleLevel
	 * @return
	 */
	final static AffineModel2D createScaleLevelTransform( final double scaleLevel )
	{
		final AffineModel2D a = new AffineModel2D();
		final double scale = Math.pow( 2, scaleLevel );
		final float t = ( float )( ( scale - 1 ) * 0.5 );
		a.set( ( float )scale, 0, 0, ( float )scale, t, t );
		return a;
	}

	
public static Object readObjectFromFileXStream(String path) {

    try{
      //filename is filepath string
      
      BufferedReader br = new BufferedReader(new FileReader(new File(path)));
            String line;
      StringBuilder sb = new StringBuilder();
      
      while((line=br.readLine())!= null){
          sb.append(line);
      }
      String xml = sb.toString();
		  br.close();
		  
	  	XStream xstream = new XStream();
	    //System.out.println(xml);
		  return xstream.fromXML(xml);
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
	public static boolean writeObjectToFile(Object o,String path) {

      //filename is filepath string
       try {
            File file = new File(path);
            file.getParentFile().mkdirs();
            Writer writer = new FileWriter(file);
            //Gson gson = new GsonBuilder().create();
            Gson gson = new GsonBuilder().setPrettyPrinting().create();
            gson.toJson(o, writer);
            writer.close();
        }
        catch ( final IOException e )
        {
            System.err.println( "Error writing JSON file: " + path );
            e.printStackTrace( System.err );
        } 
    
        catch (final Exception e){
            e.printStackTrace();
            return false;
        }
        return true;

}
}
