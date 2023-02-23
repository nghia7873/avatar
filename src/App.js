import React from 'react';
import { Button, Form, Container, Row, Col, ProgressBar } from 'react-bootstrap';
import { generateImage } from './generate.js';

import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

class App extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            uploadedImageURL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
            uploaded: false,
            fp16: 0,
            resize: "none",
            generationStatus: 0,
            updateGenerationProgressInterval: -1,
            bytesUsed: 0,
            tfBackend: "cpu"
        };
    }

    onUpload = (e) => {
        var input = e.target;
        var reader = new FileReader();
        reader.onload = () => {
            var dataURL = reader.result;
            this.setState({
                uploadedImageURL: dataURL,
                uploaded: true
            });
        };
        reader.readAsDataURL(input.files[0]);
    }

    generate = async () => {
        if (this.state.generationStatus !== 0) {
            return;
        }

        if (this.state.uploaded === false) {
            alert("Please upload an image.");
            return;
        }
        if (this.state.resize === "none") {
            alert("Please select a resize method.");
            return;
        }
        
        window.progress = 0;
        window.bytesUsed = 0;
        let updateGenerationProgressInterval = setInterval(() => {
            this.setState({
                generationProgress: window.progress * 100,
                bytesUsed: window.bytesUsed
            });

            if (this.state.generationStatus !== 1) {
                clearInterval(updateGenerationProgressInterval);
            }
        }, 500);


        this.setState({
            generationStatus: 1,
            updateGenerationProgressInterval: updateGenerationProgressInterval
        });
        let success = false;
        try {
            await generateImage(this.state.resize, this.state.fp16, "uploaded-image", "output", this.state.tfBackend);
            success = true;
        } catch (error) {
            alert("Error encountered while generating image: " + error);
            this.setState({
                generationStatus: 0
            });
        }

        if (success) {
            this.setState({
                generationStatus: 2
            });
        }
        
    }
    
    componentWillUnmount = () => {
        if (this.state.updateGenerationProgressInterval !== -1) {
            clearInterval(this.state.updateGenerationProgressInterval);
        }
    }
    
    render () {
        return (
            <div className="app">
                <Container fluid style={{"display": this.state.generationStatus === 0 ? "block" : "none"}}>
                    <Row className="margin">
                        <Col/>
                            <Col xs="12">
                                <h1 style={{"marginBottom": "20px", textAlign: "center"}}>Avatar</h1>
                            </Col>
                        <Col/>
                    </Row>
                    <Row className="margin">
                        <Col/>
                        <Col xs="12" md="8" lg="6">
                            <Form>
                                <Form.File accept="image/*" label={(this.state.uploaded ? "Change the image" : "Upload an image")} onChange={this.onUpload} multiple={false} custom />
                            </Form>
                            
                        </Col>
                        <Col/>
                    </Row>
                    <Row className="margin">
                        <Col/>
                        <Col xs="12" md="8" lg="5" xl="4" style={{textAlign: "center", margin: "20px"}}>
                            <img id="uploaded-image" alt="" src={this.state.uploadedImageURL} />
                        </Col>
                        <Col/>
                    </Row>
                    <Row className="margin">
                        <Col/>
                        <Col xs="12" md="8" lg="6" style={{textAlign: "center"}}>
                            <Form>
                                <Form.Group controlId="resize">
                                    <Form.Control defaultValue="none" as="select" onChange={(e) => this.setState({resize: e.target.value})}>
                                        <option value="none" disabled>Select Generated Image Size</option>
                                        <option value="s">Small (Fast)</option>
                                        <option value="m">Medium</option>
                                        <option value="l">Large (Slow)</option>
                                    </Form.Control>
                                </Form.Group>
                                <Form.Group controlId="tfBackend">
                                    <Form.Control as="select" onChange={(e) => this.setState({tfBackend: e.target.value})}>
                                        <option value="cpu">CPU backend for rendering (Lower Speed - For devices with weak configuration)</option>
                                        <option value="webgl">WebGL backend for rendering (100x Faster Speed - For devices with strong configuration)</option>
                                    </Form.Control>
                                </Form.Group>
                                <Button variant="primary" onClick={this.generate}>Generate</Button>
                            </Form>
                        </Col>
                        <Col/>
                    </Row>
                </Container>

                <div className="overlay" style={{"display": this.state.generationStatus === 1 ? "block" : "none"}}>
                
                    <div style={{"marginTop":"calc( 50vh - 50px )", "height": "100px", "textAlign": "center"}}>
                        <Container fluid>
                            <Row>
                                <Col/>
                                <Col xs="12" md="8" lg="6" style={{textAlign: "center"}}>
                                    <ProgressBar now={this.state.generationProgress} style={{"margin": "10px"}} />
                                    <p>Generating image...</p>
                                    <p>This may take few minutes depending on your device.</p>
                                    <p>Memory usage (MB): {this.state.bytesUsed / 1000000} </p>
                                </Col>
                                <Col/>
                            </Row>
                        </Container>
                    </div>
                    
                </div>

                <div className="overlay" style={{"display": this.state.generationStatus === 2 ? "block" : "none"}}>
                    <Container fluid>
                        <Row className="margin">
                            <Col/>
                            <Col xs="12" md="8" lg="5" xl="4" style={{textAlign: "center", margin: "20px"}}>
                                <canvas id="output"></canvas>
                            </Col>
                            <Col/>
                        </Row>
                        <Row className="margin">
                            <Col/>
                            <Col xs="12" md="12" lg="12" xl="10" style={{textAlign: "center", margin: "20px"}}>
                                <Button variant="primary" onClick={() => window.location.reload()}>Restart</Button>
                            </Col>
                            <Col/>
                        </Row>
                    </Container>
                </div>
            </div>
        );
    }
}

export default App;
