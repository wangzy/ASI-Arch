#!/bin/bash
# ASI-Arch Services Startup Script
# Starts all required backend services for ASI-Arch

set -e

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
check_docker() {
    print_info "Checking Docker..."
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Start database services
start_database() {
    print_info "Starting database services..."
    
    cd database
    
    # Start MongoDB container
    print_info "Starting MongoDB container..."
    docker-compose up -d
    
    # Wait for MongoDB to be ready
    print_info "Waiting for MongoDB to be ready..."
    sleep 10
    
    # Check if MongoDB is running
    if docker-compose ps | grep -q "mongodb.*Up"; then
        print_success "MongoDB started successfully"
    else
        print_error "MongoDB failed to start"
        docker-compose logs mongodb
        exit 1
    fi
    
    # Start API server in background
    print_info "Starting database API server..."
    nohup ./start_api.sh > ../database_api.log 2>&1 &
    DATABASE_PID=$!
    echo $DATABASE_PID > ../database_api.pid
    
    print_success "Database API server started (PID: $DATABASE_PID)"
    print_info "Database API available at: http://localhost:8001"
    print_info "Database logs: database_api.log"
    
    cd ..
}

# Start cognition base services
start_cognition_base() {
    print_info "Starting cognition base services..."
    
    cd cognition_base
    
    # Start OpenSearch container
    print_info "Starting OpenSearch container..."
    docker-compose up -d
    
    # Wait for OpenSearch to be ready
    print_info "Waiting for OpenSearch to be ready..."
    sleep 15
    
    # Start RAG API server in background
    print_info "Starting RAG API server..."
    nohup python rag_api.py > ../rag_api.log 2>&1 &
    RAG_PID=$!
    echo $RAG_PID > ../rag_api.pid
    
    print_success "RAG API server started (PID: $RAG_PID)"
    print_info "RAG API logs: rag_api.log"
    
    cd ..
}

# Wait for services to be ready
wait_for_services() {
    print_info "Waiting for services to be fully ready..."
    
    # Wait for database API
    print_info "Checking database API..."
    for i in {1..30}; do
        if curl -s http://localhost:8001/docs > /dev/null 2>&1; then
            print_success "Database API is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            print_warning "Database API may not be ready yet"
        fi
        sleep 2
    done
    
    # Wait a bit more for RAG API
    print_info "Allowing RAG API time to initialize..."
    sleep 10
    
    print_success "All services should be ready!"
}

# Display status and instructions
show_status() {
    echo ""
    echo "======================================="
    echo "ASI-Arch Services Status"
    echo "======================================="
    echo ""
    
    # Check service PIDs
    if [ -f database_api.pid ]; then
        DB_PID=$(cat database_api.pid)
        if ps -p $DB_PID > /dev/null 2>&1; then
            print_success "Database API running (PID: $DB_PID)"
        else
            print_warning "Database API may not be running"
        fi
    fi
    
    if [ -f rag_api.pid ]; then
        RAG_PID=$(cat rag_api.pid)
        if ps -p $RAG_PID > /dev/null 2>&1; then
            print_success "RAG API running (PID: $RAG_PID)"
        else
            print_warning "RAG API may not be running"
        fi
    fi
    
    echo ""
    echo "Service URLs:"
    echo "  Database API: http://localhost:8001/docs"
    echo "  RAG API: Check rag_api.log for port"
    echo ""
    echo "Log files:"
    echo "  Database API: database_api.log"
    echo "  RAG API: rag_api.log"
    echo ""
    echo "To stop services: ./stop_services.sh"
    echo "To start pipeline: cd pipeline && python pipeline.py"
    echo ""
}

# Create stop script
create_stop_script() {
    cat > stop_services.sh << 'EOF'
#!/bin/bash
# Stop ASI-Arch services

print_info() {
    echo -e "\033[0;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[0;32m[SUCCESS]\033[0m $1"
}

print_info "Stopping ASI-Arch services..."

# Stop API servers
if [ -f database_api.pid ]; then
    DB_PID=$(cat database_api.pid)
    if ps -p $DB_PID > /dev/null 2>&1; then
        kill $DB_PID
        print_success "Database API stopped"
    fi
    rm -f database_api.pid
fi

if [ -f rag_api.pid ]; then
    RAG_PID=$(cat rag_api.pid)
    if ps -p $RAG_PID > /dev/null 2>&1; then
        kill $RAG_PID
        print_success "RAG API stopped"
    fi
    rm -f rag_api.pid
fi

# Stop Docker containers
print_info "Stopping Docker containers..."
cd database && docker-compose down
cd ../cognition_base && docker-compose down
cd ..

print_success "All services stopped"
EOF
    
    chmod +x stop_services.sh
}

# Main execution
main() {
    echo "ASI-Arch Services Startup"
    echo "========================="
    echo ""
    
    # Check prerequisites
    check_docker
    
    # Start services
    start_database
    start_cognition_base
    
    # Wait for readiness
    wait_for_services
    
    # Create stop script
    create_stop_script
    
    # Show status
    show_status
}

# Run main function
main "$@"