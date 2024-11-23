from core.world.maps.map_generator import MapGenerator
from core.world.maps.map_loader import MapLoader
from core.world.maps.random_point import RandomPoint

class TrackFactory:


    @staticmethod
    def create_track(track_type, **track_args):
        """
        Creates the appropriate track based on the track type and additional parameters.

        Parameters:
        - track_type: (str) The type of the track (e.g., 'boxes', 'generator', 'random').
        - space: (object) The physics space object.
        - world_center: (tuple) The center of the world in the environment.
        - initial_pos: (tuple) The initial position for the map loader.
        - track_params: Additional parameters specific to the track type.
        """

        if track_type == "boxes":
            return MapLoader(track_args.get('boxes_file'),
                             track_args.get('sectors_file'),
                             track_args.get('initial_pos'),
                             track_args.get('rand goal'),
                             )
        elif track_type == "generator":
            return MapGenerator(track_args.get('corridor_size', 50),
                                )
        elif track_type == "simple_goal":
            return RandomPoint(spawn_range=track_args.get('spawn_range', 400),
                               fixed_goal=track_args.get('fixed_goal', [200,200]),
                               )

