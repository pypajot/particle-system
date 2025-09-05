#include "math/vec3.hpp"

#define BASE_GRAVITY 1.0f
#define MAX_GRAVITY 2.0f
#define MIN_GRAVITY 0.3f

class Gravity
{
    private:
        vec3 _pos;
        float _strength;

    public:
        bool active;
        
        Gravity();
        Gravity(vec3 pos);
        Gravity(const Gravity &other);
        ~Gravity();

        Gravity &operator=(const Gravity &other);

        void SetPos(vec3 vewPos);
        
        void GravityUp();
        void GravityDown(); 
};