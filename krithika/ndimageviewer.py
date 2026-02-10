import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, LogNorm
import itertools

class NDImageViewer:
    """Interactive viewer for 2D, 3D, and 4D image data with real-time analysis tools.

    This class provides an interactive matplotlib-based interface for viewing and analyzing
    multi-dimensional image data. It supports 2D images, 3D image cubes (e.g., time series),
    and 4D data (e.g., time series of grouped observations). Features include dynamic scaling
    modes, cut profile extraction, and draggable cut tools for pixel-level analysis.

    Parameters
    ----------
    data : array-like
        Input image data with shape (ny, nx) for 2D, (nt, ny, nx) for 3D, or
        (nt, ng, ny, nx) for 4D, where nt=time, ng=group, ny/nx=spatial dimensions.
    cmap : str, optional
        Matplotlib colormap name for image display. Default is ``'magma'``.

    Attributes
    ----------
    data : ndarray
        The input image data.
    ndim : int
        Number of dimensions in the data (2, 3, or 4).
    i_image : int
        Current image/frame index for 3D/4D data (default 0).
    i_group : int
        Current group index for 4D data (default 0).
    scale_mode : str
        Current scaling mode ('linear', 'log', 'asinh', or 'zscale').
    cuts : list of dict
        List of extracted cut profiles, each containing 'p1', 'p2', 'line', 'color',
        and 'visible' keys.
    active_cut : dict or None
        Cut currently being drawn (awaiting two clicks).
    fig : matplotlib.figure.Figure
        The interactive figure object.
    ax_img : matplotlib.axes.Axes
        Main image display axes.
    ax_prof : matplotlib.axes.Axes
        Profile plot axes for cut data.
    im : matplotlib.image.AxesImage
        The displayed image object.
    cbar : matplotlib.colorbar.Colorbar
        Colorbar for the image display.

    Notes
    -----
    - Data must be 2D, 3D, or 4D; other dimensions will raise a ValueError.
    - Finite data values are automatically detected for scaling limits.
    - The viewer is interactive: sliders update in real-time, and cuts can be drawn
      and modified by clicking/dragging on the image.

    Examples
    --------
    View a 2D image:

    >>> import numpy as np
    >>> data_2d = np.random.rand(256, 256)
    >>> viewer = NDImageViewer(data=data_2d)
    >>> viewer.show()

    View a 3D time series:

    >>> data_3d = np.random.rand(100, 256, 256)  # 100 frames
    >>> viewer = NDImageViewer(data=data_3d, cmap='viridis')
    >>> viewer.show()

    View 4D data and extract cuts:

    >>> data_4d = np.random.rand(10, 8, 256, 256)  # 10 times, 8 groups
    >>> viewer = NDImageViewer(data=data_4d)
    >>> viewer.show()
    >>> # Click "Add cut" button, then click two points on the image to define a cut
    """

    def __init__(self, data, cmap="magma"):
        """Initialize the NDImageViewer.

        Parameters
        ----------
        data : array-like
            Image data (2D, 3D, or 4D).
        cmap : str, optional
            Colormap name. Default is ``'magma'``.

        Raises
        ------
        ValueError
            If data is not 2D, 3D, or 4D.
        """

        self.data = np.asarray(data)
        self.cmap = cmap
        self.ndim = self.data.ndim

        if self.ndim not in (2, 3, 4):
            raise ValueError("Data must be 2D, 3D, or 4D")

        self.i_image = 0
        self.i_group = 0
        self.scale_mode = "linear"

        self.cuts = []
        self.active_cut = None
        self.drag_mode = None
        self.dragged_cut = None
        self.dragged_endpoint = None
        self.color_cycle = itertools.cycle(
            ["cyan", "orange", "lime", "magenta", "red", "yellow"]
        )

        self._setup_data()
        self._setup_figure()
        self._connect_events()

    # ----------------------------------------------------
    # Data helpers
    # ----------------------------------------------------
    def _setup_data(self):
        """Initialize data ranges and extract the first image for display.

        Computes finite value statistics (min/max) and initializes vmin/vmax
        for scaling. Sets up the first image frame to display.
        """
        self.img0 = self._get_image()
        self.ny, self.nx = self.img0.shape

        finite = np.isfinite(self.data)
        self.data_min = np.nanmin(self.data[finite])
        self.data_max = np.nanmax(self.data[finite])

        self.vmin = max(self.data_min, 1e-10) if self.data_min <= 0 else self.data_min
        self.vmax = self.data_max

    def _get_image(self):
        """Retrieve the current 2D image from the data based on current indices.

        Returns
        -------
        ndarray
            2D image array (shape: ny, nx).
        """
        if self.ndim == 2:
            return self.data
        elif self.ndim == 3:
            return self.data[self.i_image]
        else:
            return self.data[self.i_image, self.i_group]

    # ----------------------------------------------------
    # Figure & layout
    # ----------------------------------------------------
    def _setup_figure(self):
        """Create and configure the matplotlib figure and axes layout.

        Sets up three main components:
        - Left panel: main image display with colorbar
        - Top-right: interactive controls (sliders, scale selector)
        - Bottom-right: cut profile plot
        """
        plt.rcParams.update({
            "figure.facecolor": "#f2f2f2",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#aaaaaa",
            "axes.linewidth": 0.8,
            "font.size": 10,
        })

        self.fig = plt.figure(figsize=(12, 6))

        # ---- Left square image panel ----
        self.ax_img = self.fig.add_axes([0.05, 0.15, 0.38, 0.75])
        self.ax_img.set_xticks([])
        self.ax_img.set_yticks([])
        self.ax_img.set_title("ND Image Viewer", pad=10)

        self.im = self.ax_img.imshow(
            self.img0,
            cmap=self.cmap,
            norm=Normalize(self.vmin, self.vmax),
            origin="lower"
        )

        self.cbar = self.fig.colorbar(
            self.im,
            ax=self.ax_img,
            fraction=0.05,
            pad=0.04
        )

        # ---- Top-right controls ----
        self._setup_controls()

        # ---- Bottom-right profiles ----
        self.ax_prof = self.fig.add_axes([0.55, 0.15, 0.40, 0.35])
        self.ax_prof.set_xlabel("Pixel index")
        self.ax_prof.set_ylabel("Value", labelpad=10)

    def _setup_controls(self):
        """Create and arrange interactive control widgets.

        Widgets include:
        - vmin/vmax sliders for intensity scaling
        - Scale mode selector (Linear/Log/Asinh/Zscale)
        - Image and group sliders (for 3D/4D data)
        - "Add cut" and "Remove cut" buttons
        """
        px, pw = 0.55, 0.38
        y = 0.85
        dy = 0.055

        def label(text, ypos):
            self.fig.text(px, ypos, text, weight="semibold", color="#444")

        label("Scaling", y)
        y -= dy

        # vmin slider
        self.fig.text(px, y + 0.02, "vmin", fontsize=9)
        self.ax_vmin = self.fig.add_axes([px + 0.06, y+0.015, pw - 0.2, 0.02])
        self.s_vmin = Slider(self.ax_vmin, "", self.data_min, self.data_max, valinit=self.vmin)
        y -= dy

        # vmax slider
        self.fig.text(px, y + 0.02, "vmax", fontsize=9)
        self.ax_vmax = self.fig.add_axes([px + 0.06, y+0.015, pw - 0.2, 0.02])
        self.s_vmax = Slider(self.ax_vmax, "", self.data_min, self.data_max, valinit=self.vmax)

        
        # ------- Scale mode -------
        self.fig.text(px + 0.32, 0.85, "Scale mode", weight="semibold", color="#444")

        # Scale selector (2 columns: Linear/Log and Asinh/Zscale)
        self.ax_scale = self.fig.add_axes([px + 0.32, 0.68, 0.08, 0.15])
        self.scale_radio = RadioButtons(
            self.ax_scale,
            ["Linear", "Log", "Asinh", "Zscale"],
            active=0
        )
        for txt in self.scale_radio.labels:
            txt.set_fontsize(9)


        # ---- Navigation ----
        y -= dy * 0.8
        if self.ndim >= 3:
            label("Navigation", y)

        if self.ndim >= 3:
            y -= dy
            self.fig.text(px, y + 0.02, "Image", fontsize=9)
            self.ax_img_idx = self.fig.add_axes([px + 0.06, y+0.015, pw - 0.2, 0.02])
            self.s_img = Slider(self.ax_img_idx, "", 0, self.data.shape[0]-1,
                                valinit=0, valstep=1)

        if self.ndim == 4:
            y -= dy
            self.fig.text(px, y + 0.02, "Group", fontsize=9)
            self.ax_grp = self.fig.add_axes([px + 0.06, y+0.015, pw - 0.2, 0.02])
            self.s_grp = Slider(self.ax_grp, "", 0, self.data.shape[1]-1,
                                valinit=0, valstep=1)

        # ---- Cuts ----
        y -= dy * 1.3

        if self.ndim == 2:
            y -= 0.105     
        if self.ndim == 3:
            y -= 0.05
        if self.ndim == 4:
            y -= -0.005

        label("Cuts", y)

        #y -= dy
        self.ax_add_cut = self.fig.add_axes([px + 0.14, y-0.01, 0.12, 0.045])
        self.btn_add_cut = Button(self.ax_add_cut, "Add cut")

        self.ax_remove_cut = self.fig.add_axes([px + 0.14 + 0.14, y-0.01, 0.12, 0.045])
        self.btn_remove_cut = Button(self.ax_remove_cut, "Remove cut")

    # ----------------------------------------------------
    # Cut management
    # ----------------------------------------------------
    def _add_cut(self):
        """Initialize a new cut for the user to draw.

        Sets ``self.active_cut`` to a new dict and assigns a color from the
        color cycle. The user then clicks twice on the image to define the cut.
        """
        self.active_cut = {
            "p1": None,
            "p2": None,
            "line": None,
            "color": next(self.color_cycle),
            "visible": True
        }

    def _remove_cut(self):
        """Remove the active cut being drawn or the last finalized cut.

        If a cut is being drawn (``self.active_cut`` is not None), cancel it.
        Otherwise, pop the last cut from the ``self.cuts`` list and remove
        its line from the image.
        """
        if self.active_cut is not None:
            # Remove active cut being drawn
            self.active_cut = None
        elif self.cuts:
            # Remove the last cut
            cut = self.cuts.pop()
            if cut["line"] is not None:
                cut["line"].remove()
            self._update_profiles()

    def _finalize_cut(self):
        """Finalize a cut after two points have been clicked.

        Creates a Line2D artist on the image and appends the cut to
        ``self.cuts``. Resets ``self.active_cut`` to None and updates
        the profile plot.
        """
        cut = self.active_cut
        line = Line2D(
            [cut["p1"][0], cut["p2"][0]],
            [cut["p1"][1], cut["p2"][1]],
            lw=2,
            color=cut["color"],
            picker=5
        )
        self.ax_img.add_line(line)
        cut["line"] = line
        self.cuts.append(cut)
        self.active_cut = None
        self._update_profiles()

    def _update_profiles(self):
        """Recompute and redraw cut profiles in the profile axes.

        Clears the profile plot and redraws profiles for all visible cuts
        by sampling intensity along each cut line.
        """
        self.ax_prof.cla()
        self.ax_prof.set_xlabel("Pixel index")
        self.ax_prof.set_ylabel("Value", labelpad=10)

        img = self._get_image()

        for cut in self.cuts:
            if not cut["visible"]:
                continue
            dist, prof = self._sample_cut(img, cut["p1"], cut["p2"])
            self.ax_prof.plot(dist, prof, color=cut["color"], lw=1.8)

        self.fig.canvas.draw_idle()

    def _sample_cut(self, img, p1, p2, npts=None):
        """Sample pixel values along a line segment.

        Linearly interpolates between two points and samples the image
        at interpolated coordinates.

        Parameters
        ----------
        img : ndarray
            2D image array.
        p1 : array-like
            Starting point [x, y].
        p2 : array-like
            Ending point [x, y].
        npts : int, optional
            Number of sample points. If None, uses the pixel length of the cut.

        Returns
        -------
        dist : ndarray
            Distance along the cut (in pixels).
        values : ndarray
            Sampled intensity values along the line.
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if npts is None:
            npts = int(max(abs(dx), abs(dy))) + 1
            npts = max(npts, 2)

        x = np.linspace(p1[0], p2[0], npts)
        y = np.linspace(p1[1], p2[1], npts)
        xi = np.clip(np.round(x).astype(int), 0, self.nx - 1)
        yi = np.clip(np.round(y).astype(int), 0, self.ny - 1)
        dist = np.sqrt((x - p1[0]) ** 2 + (y - p1[1]) ** 2)
        return dist, img[yi, xi]

    def _get_distance_to_point(self, p1, p2, threshold=10):
        """Compute distance between two points and check if within threshold.

        Parameters
        ----------
        p1 : array-like
            First point [x, y].
        p2 : array-like
            Second point [x, y].
        threshold : float, optional
            Distance threshold in pixels. Default is 10.

        Returns
        -------
        endpoint : str or None
            'p1' or 'p2' if within threshold, otherwise None.
        distance : float
            Computed distance, or None if not within threshold.
        """
        dist_p1 = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        dist_p2 = np.sqrt((self.data.shape[-1] - p2[0])**2 + (self.data.shape[-2] - p2[1])**2)
        
        if dist_p1 < threshold:
            return "p1", dist_p1
        elif dist_p2 < threshold:
            return "p2", dist_p2
        return None, None

    def _get_distance_to_line(self, p1, p2, point, threshold=10):
        """Compute perpendicular distance from a point to a line segment.

        Parameters
        ----------
        p1 : array-like
            Line segment start [x, y].
        p2 : array-like
            Line segment end [x, y].
        point : array-like
            Query point [x, y].
        threshold : float, optional
            Distance threshold in pixels. Default is 10.

        Returns
        -------
        distance : float
            Perpendicular distance to the line segment, or ``inf`` if the
            perpendicular from the point does not intersect the segment.
        """
        x1, y1 = p1
        x2, y2 = p2
        x0, y0 = point

        # Distance from point to line segment
        num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        den = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        if den == 0:
            return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        dist = num / den

        # Check if point projects onto the segment
        t = ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)) / (den**2)
        if 0 <= t <= 1 and dist < threshold:
            return dist
        return float('inf')

    # ----------------------------------------------------
    # Events
    # ----------------------------------------------------
    def _connect_events(self):
        """Connect matplotlib event handlers for interactivity.

        Connects slider callbacks, button click handlers, and mouse event
        handlers for image interaction.
        """
        self.s_vmin.on_changed(self._update_image)
        self.s_vmax.on_changed(self._update_image)

        if self.ndim >= 3:
            self.s_img.on_changed(self._update_image)
        if self.ndim == 4:
            self.s_grp.on_changed(self._update_image)

        self.scale_radio.on_clicked(self._change_scale)
        self.btn_add_cut.on_clicked(lambda _: self._add_cut())
        self.btn_remove_cut.on_clicked(lambda _: self._remove_cut())

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_drag)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("pick_event", self._on_pick)

    def _update_image(self, val=None):
        """Update the displayed image and rescale it based on current settings.

        Called when sliders change or scale mode is switched. Fetches the
        current image, applies the selected scaling mode, updates the display,
        and recomputes cut profiles.

        Parameters
        ----------
        val : float, optional
            Value from slider (unused, present for callback compatibility).
        """
        if self.ndim >= 3:
            self.i_image = int(self.s_img.val)
        if self.ndim == 4:
            self.i_group = int(self.s_grp.val)

        img = self._get_image()

        if self.scale_mode == "log":
            pos = img[img > 0]
            norm = LogNorm(max(self.s_vmin.val, pos.min() if len(pos) > 0 else 1e-10), self.s_vmax.val)
        elif self.scale_mode == "asinh":
            from matplotlib.colors import SymLogNorm
            norm = SymLogNorm(linthresh=0.03, vmin=self.s_vmin.val, vmax=self.s_vmax.val)
        elif self.scale_mode == "zscale":
            # Simple zscale-like normalization: use percentiles
            vmin_z = np.percentile(img[np.isfinite(img)], 2)
            vmax_z = np.percentile(img[np.isfinite(img)], 98)
            norm = Normalize(vmin_z, vmax_z)
        else:
            norm = Normalize(self.s_vmin.val, self.s_vmax.val)

        self.im.set_data(img)
        self.im.set_norm(norm)
        self.cbar.update_normal(self.im)

        self._update_profiles()

    def _change_scale(self, label):
        """Switch the scaling mode and update the image.

        Parameters
        ----------
        label : str
            Scale mode name ('Linear', 'Log', 'Asinh', or 'Zscale').
        """
        self.scale_mode = label.lower()
        self._update_image()

    def _on_click(self, event):
        """Handle mouse button press events on the image.

        Supports:
        - Drawing cuts (two clicks to define start and end points)
        - Selecting cut endpoints or lines for dragging

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Mouse event with xdata, ydata, and inaxes attributes.
        """
        if event.inaxes != self.ax_img:
            return

        # If drawing a new cut
        if self.active_cut is not None:
            if self.active_cut["p1"] is None:
                self.active_cut["p1"] = [event.xdata, event.ydata]
            else:
                self.active_cut["p2"] = [event.xdata, event.ydata]
                self._finalize_cut()
            return

        # Check if clicking near an existing cut endpoint or line
        point = [event.xdata, event.ydata]
        for i, cut in enumerate(self.cuts):
            # Check distance to endpoints
            endpoint, dist = self._get_distance_to_point(cut["p1"], point, threshold=10)
            if endpoint is not None:
                self.dragged_cut = i
                self.dragged_endpoint = endpoint
                self.drag_mode = "endpoint"
                return

            endpoint2, dist2 = self._get_distance_to_point(cut["p2"], point, threshold=10)
            if endpoint2 is not None:
                self.dragged_cut = i
                self.dragged_endpoint = endpoint2
                self.drag_mode = "endpoint"
                return

            # Check distance to line
            line_dist = self._get_distance_to_line(cut["p1"], cut["p2"], point, threshold=10)
            if line_dist < 10:
                self.dragged_cut = i
                self.drag_mode = "line"
                self.drag_start = point
                return

    def _on_drag(self, event):
        """Handle mouse drag events for moving cuts or endpoints.

        Allows dragging individual endpoints or translating entire cuts.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Mouse motion event.
        """
        if event.inaxes != self.ax_img or self.drag_mode is None or self.dragged_cut is None:
            return

        cut = self.cuts[self.dragged_cut]
        point = [event.xdata, event.ydata]

        if self.drag_mode == "endpoint":
            if self.dragged_endpoint == "p1":
                cut["p1"] = point
            else:
                cut["p2"] = point
            # Update line
            cut["line"].set_data([cut["p1"][0], cut["p2"][0]], [cut["p1"][1], cut["p2"][1]])
            self._update_profiles()

        elif self.drag_mode == "line":
            # Translate both endpoints
            dx = point[0] - self.drag_start[0]
            dy = point[1] - self.drag_start[1]
            cut["p1"] = [cut["p1"][0] + dx, cut["p1"][1] + dy]
            cut["p2"] = [cut["p2"][0] + dx, cut["p2"][1] + dy]
            cut["line"].set_data([cut["p1"][0], cut["p2"][0]], [cut["p1"][1], cut["p2"][1]])
            self.drag_start = point
            self._update_profiles()

    def _on_release(self, event):
        """Handle mouse button release events to end dragging.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Mouse release event.
        """
        self.drag_mode = None
        self.dragged_cut = None
        self.dragged_endpoint = None

    def _on_pick(self, event):
        """Handle pick events on cut lines (currently unused).

        Parameters
        ----------
        event : matplotlib.backend_bases.PickEvent
            Pick event (fires when artist is clicked with appropriate picker tolerance).
        """
        pass

    # ----------------------------------------------------
    def show(self):
        """Display the interactive viewer window.

        Calls ``plt.show()`` to render the figure and start the interactive loop.
        """
        plt.show()