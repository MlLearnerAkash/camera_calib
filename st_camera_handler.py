import stapipy as st
import cv2
import numpy as np

# Global initialization - happens ONCE
_st_initialized = False
_st_system = None
_all_devices = []
_all_datastreams = []

def _initialize_st_once():
    """Initialize StApi and discover all cameras - called ONCE"""
    global _st_initialized, _st_system, _all_devices, _all_datastreams
    
    if _st_initialized:
        return  # Already initialized
    
    try:
        print("Initializing StApi (Global)...")
        st.initialize()
        
        # Create system for camera discovery
        _st_system = st.create_system()
        
        # Discover ALL available cameras
        print("Discovering cameras...")
        while True:
            try:
                device = _st_system.create_first_device()
                datastream = device.create_datastream()
                _all_devices.append(device)
                _all_datastreams.append(datastream)
                print(f"  ✓ Camera {len(_all_devices)}: {device.info.display_name}")
            except:
                break
        
        if len(_all_devices) == 0:
            raise RuntimeError("No StApi cameras found!")
        
        _st_initialized = True
        print(f"✓ StApi initialized with {len(_all_devices)} camera(s)\n")
        
    except Exception as e:
        print(f"✗ StApi initialization failed: {str(e)}")
        raise


class StCameraHandler:
    """Wrapper for individual StApi camera - uses global initialized system"""
    
    def __init__(self, camera_index=0):
        """
        Initialize a single StApi camera handler
        
        Args:
            camera_index: Index of camera to use (0 for first, 1 for second, etc.)
        """
        # Initialize StApi globally (only happens once)
        _initialize_st_once()
        
        if camera_index >= len(_all_devices):
            raise ValueError(
                f"Camera index {camera_index} not available. "
                f"Found {len(_all_devices)} cameras."
            )
        
        # Reference the pre-initialized device and datastream
        self.camera_index = camera_index
        self.device = _all_devices[camera_index]
        self.datastream = _all_datastreams[camera_index]
        
        # Setup converter for image format
        self.converter = st.create_converter(st.EStConverterType.PixelFormat)
        self.converter.destination_pixel_format = st.EStPixelFormatNamingConvention.BGR8
        
        self.current_frame = None
        self.is_opened = True
        self.is_grabbing = False
        
        print(f"✓ Camera {camera_index} handler created: {self.device.info.display_name}")
    
    def read(self):
        """
        Read a frame from the camera - matches cv2.VideoCapture.read() interface
        
        Returns:
            (ret, frame): ret=True if successful, frame=numpy array BGR image
        """
        try:
            # Start acquisition on first read
            if not self.is_grabbing:
                self.datastream.start_acquisition(1)
                self.device.acquisition_start()
                self.is_grabbing = True
                print(f"  → Acquisition started for Camera {self.camera_index}")
            
            # Retrieve buffer with timeout
            with self.datastream.retrieve_buffer(timeout=5000) as st_buffer:
                if st_buffer.info.is_image_present:
                    st_image = st_buffer.get_image()
                    st_image = self.converter.convert(st_image)
                    
                    # Convert to numpy array
                    data = st_image.get_image_data()
                    nparr = np.frombuffer(data, np.uint8).copy()
                    frame = nparr.reshape(st_image.height, st_image.width, 3).copy()
                    
                    self.current_frame = frame
                    return True, frame
                else:
                    return False, None
                    
        except Exception as e:
            print(f"Error reading from Camera {self.camera_index}: {str(e)}")
            return False, None
    
    def isOpened(self):
        """Check if camera is opened"""
        return self.is_opened
    
    def release(self):
        """Release camera resources"""
        try:
            if self.is_grabbing:
                self.datastream.stop_acquisition()
                self.device.acquisition_stop()
                self.is_grabbing = False
            self.is_opened = False
            print(f"✓ Camera {self.camera_index} released")
        except Exception as e:
            print(f"Error releasing Camera {self.camera_index}: {e}")


def cleanup_st():
    """Call this when closing the application"""
    global _st_initialized
    try:
        for device in _all_devices:
            device.acquisition_stop()
        for datastream in _all_datastreams:
            if datastream.is_grabbing:
                datastream.stop_acquisition()
        st.terminate()
        _st_initialized = False
        print("✓ StApi cleaned up")
    except Exception as e:
        print(f"Error cleaning up StApi: {e}")